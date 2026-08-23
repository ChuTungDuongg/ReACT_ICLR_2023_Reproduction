"""Act-only agent that alternates actions and Wikipedia observations."""

from __future__ import annotations

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import ActionParseError, parse_tool_action
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.hotpotqa import build_act_prompt
from react_reproduction.tools.wikipedia import ActionType, WikipediaEnvironment


class ActOnlyAgent(BaseAgent):
    def __init__(
        self,
        llm: LLMProvider,
        generation: GenerationConfig,
        environment: WikipediaEnvironment,
        *,
        max_steps: int,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._llm = llm
        self._generation = generation
        self._environment = environment
        self._max_steps = max_steps

    def predict(self, example: BenchmarkExample) -> AgentResult:
        self._environment.reset()
        trajectory: list[TrajectoryStep] = []
        tool_calls = 0
        last_parse_failed = False
        pending_termination_reason: str | None = None

        for step_index in range(1, self._max_steps + 1):
            force_finish = (
                pending_termination_reason is not None
                or step_index == self._max_steps
            )
            model_output = self._llm.generate(
                build_act_prompt(
                    example.input_text,
                    trajectory,
                    force_finish=force_finish,
                ),
                temperature=self._generation.temperature,
                top_p=self._generation.top_p,
                max_new_tokens=self._generation.max_new_tokens,
            )
            try:
                action = parse_tool_action(model_output)
            except ActionParseError as error:
                last_parse_failed = True
                observation = f"Invalid action format: {error}"
                trajectory.append(
                    TrajectoryStep(
                        step_index=step_index,
                        model_output=model_output,
                        observation=observation,
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
                if step_index < self._max_steps:
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
