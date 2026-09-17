"""ReAct agent interleaving explicit reasoning with Wikipedia actions."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import (
    ActionParseError,
    parse_react_output,
    parse_thought,
    parse_tool_action,
)
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import (
    LLMProvider,
    generate_batch_with_stops,
    truncate_at_stop_sequences,
)
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


@dataclass(frozen=True, slots=True)
class _ReActTurn:
    model_output: str
    thought: str | None
    action: ToolAction | None = None
    parse_error: ActionParseError | None = None


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
        prompt_builder: Callable[..., str] = build_react_prompt,
        answer_normalizer: Callable[[str], str] = str.strip,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._llm = llm
        self._generation = generation
        self._environment = environment
        self._max_steps = max_steps
        self._best_effort_finalization = best_effort_finalization
        self._environment_factory = environment_factory
        self._prompt_builder = prompt_builder
        self._answer_normalizer = answer_normalizer
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
                self._prompt_builder(
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
            model_outputs = generate_batch_with_stops(
                self._llm,
                prompts,
                temperature=self._generation.temperature,
                top_p=self._generation.top_p,
                max_new_tokens=self._generation.max_new_tokens,
                stop_sequences=(f"\nObservation {step_index}:",),
            )
            if len(model_outputs) != len(active_states):
                raise RuntimeError(
                    "LLM batch output count does not match active ReAct states: "
                    f"{len(model_outputs)} != {len(active_states)}."
                )
            turns: list[_ReActTurn] = []
            recovery_indexes: list[int] = []
            for active_index, model_output in enumerate(model_outputs):
                model_output = truncate_at_stop_sequences(
                    model_output,
                    (f"\nObservation {step_index}:",),
                )
                try:
                    thought, action = parse_react_output(model_output)
                except ActionParseError as error:
                    turns.append(
                        _ReActTurn(
                            model_output=model_output,
                            thought=self._recover_thought(model_output),
                            parse_error=error,
                        )
                    )
                    recovery_indexes.append(active_index)
                else:
                    turns.append(
                        _ReActTurn(
                            model_output=model_output,
                            thought=thought,
                            action=action,
                        )
                    )

            if recovery_indexes:
                recovery_prompts = [
                    self._build_action_recovery_prompt(
                        prompts[index],
                        turns[index].thought,
                        step_index,
                    )
                    for index in recovery_indexes
                ]
                recovery_outputs = generate_batch_with_stops(
                    self._llm,
                    recovery_prompts,
                    temperature=self._generation.temperature,
                    top_p=self._generation.top_p,
                    max_new_tokens=self._generation.max_new_tokens,
                    stop_sequences=("\n",),
                )
                if len(recovery_outputs) != len(recovery_indexes):
                    raise RuntimeError(
                        "LLM batch output count does not match ReAct recovery "
                        f"states: {len(recovery_outputs)} != "
                        f"{len(recovery_indexes)}."
                    )
                for active_index, recovery_output in zip(
                    recovery_indexes,
                    recovery_outputs,
                    strict=True,
                ):
                    recovery_output = truncate_at_stop_sequences(
                        recovery_output,
                        ("\n",),
                    )
                    try:
                        action = parse_tool_action(recovery_output)
                    except ActionParseError as error:
                        turns[active_index] = _ReActTurn(
                            model_output=recovery_output,
                            thought=turns[active_index].thought,
                            parse_error=error,
                        )
                    else:
                        turns[active_index] = _ReActTurn(
                            model_output=recovery_output,
                            thought=turns[active_index].thought,
                            action=action,
                        )

            for state, turn, force_finish in zip(
                active_states,
                turns,
                force_finish_flags,
                strict=True,
            ):
                self._advance_state(
                    state,
                    turn,
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
        turn: _ReActTurn,
        *,
        step_index: int,
        force_finish: bool,
    ) -> None:
        if turn.parse_error is not None:
            state.last_parse_failed = True
            state.trajectory.append(
                TrajectoryStep(
                    step_index=step_index,
                    model_output=turn.model_output,
                    thought=turn.thought,
                    observation=f"Invalid action format: {turn.parse_error}",
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

        if turn.action is None:
            raise RuntimeError("Parsed ReAct turn is missing its action.")
        state.last_parse_failed = False
        if force_finish:
            self._finish_forced_step(
                state,
                turn.model_output,
                turn.thought,
                turn.action,
                step_index,
            )
            return

        execution = state.environment.execute(turn.action)
        state.tool_calls += int(execution.tool_called)
        state.trajectory.append(
            TrajectoryStep(
                step_index=step_index,
                model_output=turn.model_output,
                thought=turn.thought,
                action=turn.action.canonical,
                observation=execution.observation,
            )
        )
        if not execution.terminated:
            return
        if execution.answer is not None:
            prediction = self._answer_normalizer(execution.answer)
            state.result = AgentResult(
                prediction=prediction,
                steps=len(state.trajectory),
                tool_calls=state.tool_calls,
                termination_reason=(
                    execution.termination_reason or "terminated"
                    if prediction
                    else "invalid_label"
                ),
                trajectory=tuple(state.trajectory),
                metadata=(
                    {}
                    if prediction
                    else {
                        "raw_prediction": execution.answer,
                        "parse_error": "Finish did not contain a valid task answer.",
                    }
                ),
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
    def _recover_thought(model_output: str) -> str | None:
        """Recover the first usable reasoning line, matching the paper code."""
        explicit = parse_thought(model_output)
        candidates = explicit.splitlines() if explicit else model_output.splitlines()
        for line in candidates:
            candidate = line.strip()
            if not candidate:
                continue
            if re.match(
                r"^(?:action|observation)(?:\s+\d+)?\s*:",
                candidate,
                flags=re.IGNORECASE,
            ):
                return None
            return candidate
        return None

    @staticmethod
    def _build_action_recovery_prompt(
        prompt: str,
        thought: str | None,
        step_index: int,
    ) -> str:
        thought_label = f"Thought {step_index}:"
        if re.search(rf"{re.escape(thought_label)}\s*$", prompt):
            rendered_thought = f" {thought}" if thought else ""
            return f"{prompt}{rendered_thought}\nAction {step_index}:"
        rendered_thought = f" {thought}" if thought else ""
        return (
            f"{prompt}\n{thought_label}{rendered_thought}\n"
            f"Action {step_index}:"
        )

    def _finish_forced_step(
        self,
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

        raw_answer = action.argument.strip()
        answer = self._answer_normalizer(raw_answer)
        state.trajectory.append(
            TrajectoryStep(
                step_index=step_index,
                model_output=model_output,
                thought=thought,
                action=action.canonical,
                observation=f"Finished with best-effort answer: {raw_answer}",
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
            termination_reason=termination_reason if answer else "invalid_label",
            trajectory=tuple(state.trajectory),
            metadata=(
                {}
                if answer
                else {
                    "raw_prediction": raw_answer,
                    "parse_error": "Finish did not contain a valid task answer.",
                }
            ),
        )
