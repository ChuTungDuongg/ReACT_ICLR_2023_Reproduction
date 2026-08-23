"""ReAct agent interleaving explicit reasoning with Wikipedia actions."""

from __future__ import annotations

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import (
    ActionParseError,
    parse_react_output,
    parse_thought,
)
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.hotpotqa import build_react_prompt
from react_reproduction.tools.wikipedia import ActionType, WikipediaEnvironment


class ReActAgent(BaseAgent):
    """Run the Thought -> Action -> Observation loop for one question."""

    def __init__(
        self,
        llm: LLMProvider,
        generation: GenerationConfig,
        environment: WikipediaEnvironment,
        *,
        max_steps: int,
        best_effort_finalization: bool = True,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._llm = llm
        self._generation = generation
        self._environment = environment
        self._max_steps = max_steps
        self._best_effort_finalization = best_effort_finalization

    def predict(self, example: BenchmarkExample) -> AgentResult:
        self._environment.reset()
        trajectory: list[TrajectoryStep] = []
        tool_calls = 0
        last_parse_failed = False
        pending_termination_reason: str | None = None

        for step_index in range(1, self._max_steps + 1):
            force_finish = (
                self._best_effort_finalization
                and (
                    pending_termination_reason is not None
                    or step_index == self._max_steps
                )
            )
            model_output = self._llm.generate(
                build_react_prompt(
                    example.input_text,
                    trajectory,
                    force_finish=force_finish,
                ),
                temperature=self._generation.temperature,
                top_p=self._generation.top_p,
                max_new_tokens=self._generation.max_new_tokens,
            )
            try:
                thought, action = parse_react_output(model_output)
            except ActionParseError as error:
                last_parse_failed = True
                trajectory.append(
                    TrajectoryStep(
                        step_index=step_index,
                        model_output=model_output,
                        thought=parse_thought(model_output),
                        observation=f"Invalid action format: {error}",
                    )
                )
                if force_finish:
                    return AgentResult(
                        prediction="",
                        steps=len(trajectory),
                        tool_calls=tool_calls,
                        termination_reason="finalization_failed",
                        trajectory=tuple(trajectory),
                    )
                continue

            last_parse_failed = False
            if force_finish:
                if action.action_type is not ActionType.FINISH:
                    trajectory.append(
                        TrajectoryStep(
                            step_index=step_index,
                            model_output=model_output,
                            thought=thought,
                            action=action.canonical,
                            observation=(
                                "Final step requires Finish[answer]; the tool "
                                "action was not executed."
                            ),
                        )
                    )
                    return AgentResult(
                        prediction="",
                        steps=len(trajectory),
                        tool_calls=tool_calls,
                        termination_reason="finalization_failed",
                        trajectory=tuple(trajectory),
                    )

                answer = action.argument.strip()
                trajectory.append(
                    TrajectoryStep(
                        step_index=step_index,
                        model_output=model_output,
                        thought=thought,
                        action=action.canonical,
                        observation=f"Finished with best-effort answer: {answer}",
                    )
                )
                termination_reason = (
                    f"completed_after_{pending_termination_reason}"
                    if pending_termination_reason is not None
                    else "completed_at_step_limit"
                )
                return AgentResult(
                    prediction=answer,
                    steps=len(trajectory),
                    tool_calls=tool_calls,
                    termination_reason=termination_reason,
                    trajectory=tuple(trajectory),
                )

            execution = self._environment.execute(action)
            tool_calls += int(execution.tool_called)
            trajectory.append(
                TrajectoryStep(
                    step_index=step_index,
                    model_output=model_output,
                    thought=thought,
                    action=action.canonical,
                    observation=execution.observation,
                )
            )
            if execution.terminated:
                if execution.answer is not None:
                    return AgentResult(
                        prediction=execution.answer,
                        steps=len(trajectory),
                        tool_calls=tool_calls,
                        termination_reason=(
                            execution.termination_reason or "terminated"
                        ),
                        trajectory=tuple(trajectory),
                    )
                if self._best_effort_finalization and step_index < self._max_steps:
                    pending_termination_reason = (
                        execution.termination_reason or "terminated"
                    )
                    continue
                return AgentResult(
                    prediction="",
                    steps=len(trajectory),
                    tool_calls=tool_calls,
                    termination_reason=(
                        execution.termination_reason or "terminated"
                    ),
                    trajectory=tuple(trajectory),
                )

        return AgentResult(
            prediction="",
            steps=len(trajectory),
            tool_calls=tool_calls,
            termination_reason=(
                "parsing_error" if last_parse_failed else "max_steps_exceeded"
            ),
            trajectory=tuple(trajectory),
        )
