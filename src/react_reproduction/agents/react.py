"""ReAct agent interleaving explicit reasoning with Wikipedia actions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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
from react_reproduction.tools.wikipedia import (
    ActionType,
    ToolAction,
    WikipediaEnvironment,
)


@dataclass(slots=True)
class _ReActState:
    example: BenchmarkExample
    environment: WikipediaEnvironment
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    tool_calls: int = 0
    last_parse_failed: bool = False
    pending_termination_reason: str | None = None
    result: AgentResult | None = None


class ReActAgent(BaseAgent):
    """Run Thought -> Action -> Observation for one or more questions."""

    def __init__(
        self,
        llm: LLMProvider,
        generation: GenerationConfig,
        environment: WikipediaEnvironment,
        *,
        max_steps: int,
        best_effort_finalization: bool = False,
        environment_factory: Callable[[], WikipediaEnvironment] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._llm = llm
        self._generation = generation
        self._environment = environment
        self._max_steps = max_steps
        self._best_effort_finalization = best_effort_finalization
        self._environment_factory = environment_factory
        self._batch_environments = [environment]

    def predict(self, example: BenchmarkExample) -> AgentResult:
        return self._predict_with_environments((example,), (self._environment,))[0]

    def predict_batch(
        self,
        examples: Sequence[BenchmarkExample],
    ) -> tuple[AgentResult, ...]:
        if not examples:
            return ()
        if len(examples) == 1:
            return (self.predict(examples[0]),)
        if self._environment_factory is None:
            return super().predict_batch(examples)
        while len(self._batch_environments) < len(examples):
            self._batch_environments.append(self._environment_factory())
        environments = tuple(self._batch_environments[: len(examples)])
        return self._predict_with_environments(examples, environments)

    def _predict_with_environments(
        self,
        examples: Sequence[BenchmarkExample],
        environments: Sequence[WikipediaEnvironment],
    ) -> tuple[AgentResult, ...]:
        if len(examples) != len(environments):
            raise ValueError("Each ReAct example requires its own environment.")
        states = [
            _ReActState(example=example, environment=environment)
            for example, environment in zip(examples, environments, strict=True)
        ]
        for state in states:
            state.environment.reset()

        for step_index in range(1, self._max_steps + 1):
            active_states = [state for state in states if state.result is None]
            if not active_states:
                break
            force_finish_flags = [
                self._best_effort_finalization
                and (
                    state.pending_termination_reason is not None
                    or step_index == self._max_steps
                )
                for state in active_states
            ]
            prompts = [
                build_react_prompt(
                    state.example.input_text,
                    state.trajectory,
                    force_finish=force_finish,
                )
                for state, force_finish in zip(
                    active_states,
                    force_finish_flags,
                    strict=True,
                )
            ]
            model_outputs = self._llm.generate_batch(
                prompts,
                temperature=self._generation.temperature,
                top_p=self._generation.top_p,
                max_new_tokens=self._generation.max_new_tokens,
            )
            if len(model_outputs) != len(active_states):
                raise RuntimeError(
                    "LLM batch output count does not match active ReAct states: "
                    f"{len(model_outputs)} != {len(active_states)}."
                )
            for state, model_output, force_finish in zip(
                active_states,
                model_outputs,
                force_finish_flags,
                strict=True,
            ):
                self._advance_state(
                    state,
                    model_output,
                    step_index=step_index,
                    force_finish=force_finish,
                )

        for state in states:
            if state.result is None:
                state.result = AgentResult(
                    prediction="",
                    steps=len(state.trajectory),
                    tool_calls=state.tool_calls,
                    termination_reason=(
                        "parsing_error"
                        if state.last_parse_failed
                        else "max_steps_exceeded"
                    ),
                    trajectory=tuple(state.trajectory),
                )
        results: list[AgentResult] = []
        for state in states:
            if state.result is None:
                raise RuntimeError("ReAct batch state did not produce a result.")
            results.append(state.result)
        return tuple(results)

    def _advance_state(
        self,
        state: _ReActState,
        model_output: str,
        *,
        step_index: int,
        force_finish: bool,
    ) -> None:
        try:
            thought, action = parse_react_output(model_output)
        except ActionParseError as error:
            state.last_parse_failed = True
            state.trajectory.append(
                TrajectoryStep(
                    step_index=step_index,
                    model_output=model_output,
                    thought=parse_thought(model_output),
                    observation=f"Invalid action format: {error}",
                )
            )
            if force_finish:
                state.result = AgentResult(
                    prediction="",
                    steps=len(state.trajectory),
                    tool_calls=state.tool_calls,
                    termination_reason="finalization_failed",
                    trajectory=tuple(state.trajectory),
                )
            return

        state.last_parse_failed = False
        if force_finish:
            self._finish_forced_step(state, model_output, thought, action, step_index)
            return

        execution = state.environment.execute(action)
        state.tool_calls += int(execution.tool_called)
        state.trajectory.append(
            TrajectoryStep(
                step_index=step_index,
                model_output=model_output,
                thought=thought,
                action=action.canonical,
                observation=execution.observation,
            )
        )
        if not execution.terminated:
            return
        if execution.answer is not None:
            state.result = AgentResult(
                prediction=execution.answer,
                steps=len(state.trajectory),
                tool_calls=state.tool_calls,
                termination_reason=(execution.termination_reason or "terminated"),
                trajectory=tuple(state.trajectory),
            )
            return
        if self._best_effort_finalization and step_index < self._max_steps:
            state.pending_termination_reason = (
                execution.termination_reason or "terminated"
            )
            return
        state.result = AgentResult(
            prediction="",
            steps=len(state.trajectory),
            tool_calls=state.tool_calls,
            termination_reason=(execution.termination_reason or "terminated"),
            trajectory=tuple(state.trajectory),
        )

    @staticmethod
    def _finish_forced_step(
        state: _ReActState,
        model_output: str,
        thought: str,
        action: ToolAction,
        step_index: int,
    ) -> None:
        if action.action_type is not ActionType.FINISH:
            state.trajectory.append(
                TrajectoryStep(
                    step_index=step_index,
                    model_output=model_output,
                    thought=thought,
                    action=action.canonical,
                    observation=(
                        "Final step requires Finish[answer]; the tool action "
                        "was not executed."
                    ),
                )
            )
            state.result = AgentResult(
                prediction="",
                steps=len(state.trajectory),
                tool_calls=state.tool_calls,
                termination_reason="finalization_failed",
                trajectory=tuple(state.trajectory),
            )
            return

        answer = action.argument.strip()
        state.trajectory.append(
            TrajectoryStep(
                step_index=step_index,
                model_output=model_output,
                thought=thought,
                action=action.canonical,
                observation=f"Finished with best-effort answer: {answer}",
            )
        )
        termination_reason = (
            f"completed_after_{state.pending_termination_reason}"
            if state.pending_termination_reason is not None
            else "completed_at_step_limit"
        )
        state.result = AgentResult(
            prediction=answer,
            steps=len(state.trajectory),
            tool_calls=state.tool_calls,
            termination_reason=termination_reason,
            trajectory=tuple(state.trajectory),
        )
