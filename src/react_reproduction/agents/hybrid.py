"""Paper-style fallback policies combining ReAct and CoT-SC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Mapping

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.cot_sc import CoTSCAgent, CoTSCOutcome
from react_reproduction.datasets.base import BenchmarkExample


class ReActThenCoTSCAgent(BaseAgent):
    """Use ReAct first and fall back to CoT-SC when ReAct has no answer."""

    def __init__(self, react: BaseAgent, cot_sc: CoTSCAgent) -> None:
        self._react = react
        self._cot_sc = cot_sc

    def predict(self, example: BenchmarkExample) -> AgentResult:
        return self.predict_batch((example,))[0]

    def predict_batch(
        self,
        examples: Sequence[BenchmarkExample],
    ) -> tuple[AgentResult, ...]:
        if not examples:
            return ()
        react_results = self._react.predict_batch(examples)
        if len(react_results) != len(examples):
            raise RuntimeError("ReAct batch output count does not match input count.")
        fallback_indices = [
            index
            for index, result in enumerate(react_results)
            if not (result.prediction and result.termination_reason == "completed")
        ]
        fallback_outcomes = self._cot_sc.predict_batch_with_consensus(
            [examples[index] for index in fallback_indices]
        )
        outcomes_by_index = dict(zip(fallback_indices, fallback_outcomes, strict=True))
        return tuple(
            self._combine_result(react_result, outcomes_by_index.get(index))
            for index, react_result in enumerate(react_results)
        )

    @staticmethod
    def _combine_result(
        react_result: AgentResult,
        cot_outcome: CoTSCOutcome | None,
    ) -> AgentResult:
        react_steps = _phase_steps(react_result.trajectory, "react")
        if react_result.prediction and react_result.termination_reason == "completed":
            return AgentResult(
                prediction=react_result.prediction,
                steps=len(react_steps),
                tool_calls=react_result.tool_calls,
                termination_reason="react_completed",
                trajectory=react_steps,
                supporting_facts=react_result.supporting_facts,
                metadata={
                    "selected_path": "react",
                    "fallback_used": False,
                    "react_termination_reason": react_result.termination_reason,
                },
            )

        if cot_outcome is None:
            raise RuntimeError("Missing CoT-SC fallback output for failed ReAct result.")
        cot_steps = _phase_steps(
            cot_outcome.result.trajectory,
            "cot_sc",
            start_index=len(react_steps) + 1,
        )
        prediction = cot_outcome.result.prediction
        metadata = _hybrid_metadata(
            cot_outcome,
            selected_path="cot_sc",
            fallback_used=True,
            extra={"react_termination_reason": react_result.termination_reason},
        )
        return AgentResult(
            prediction=prediction,
            steps=len(react_steps) + len(cot_steps),
            tool_calls=react_result.tool_calls,
            termination_reason=(
                "fallback_cot_sc" if prediction else "hybrid_failed"
            ),
            trajectory=react_steps + cot_steps,
            metadata=metadata,
        )


class CoTSCThenReActAgent(BaseAgent):
    """Use CoT-SC first and fall back to ReAct on low vote confidence."""

    def __init__(self, cot_sc: CoTSCAgent, react: BaseAgent) -> None:
        self._cot_sc = cot_sc
        self._react = react

    def predict(self, example: BenchmarkExample) -> AgentResult:
        return self.predict_batch((example,))[0]

    def predict_batch(
        self,
        examples: Sequence[BenchmarkExample],
    ) -> tuple[AgentResult, ...]:
        if not examples:
            return ()
        cot_outcomes = self._cot_sc.predict_batch_with_consensus(examples)
        fallback_indices = [
            index
            for index, outcome in enumerate(cot_outcomes)
            if not (outcome.result.prediction and outcome.meets_paper_threshold)
        ]
        fallback_results = self._react.predict_batch(
            [examples[index] for index in fallback_indices]
        )
        react_by_index = dict(zip(fallback_indices, fallback_results, strict=True))
        return tuple(
            self._combine_result(cot_outcome, react_by_index.get(index))
            for index, cot_outcome in enumerate(cot_outcomes)
        )

    @staticmethod
    def _combine_result(
        cot_outcome: CoTSCOutcome,
        react_result: AgentResult | None,
    ) -> AgentResult:
        cot_steps = _phase_steps(cot_outcome.result.trajectory, "cot_sc")
        if cot_outcome.result.prediction and cot_outcome.meets_paper_threshold:
            return AgentResult(
                prediction=cot_outcome.result.prediction,
                steps=len(cot_steps),
                tool_calls=0,
                termination_reason="cot_sc_consensus",
                trajectory=cot_steps,
                metadata=_hybrid_metadata(
                    cot_outcome,
                    selected_path="cot_sc",
                    fallback_used=False,
                ),
            )

        if react_result is None:
            raise RuntimeError("Missing ReAct fallback output for low-confidence CoT-SC.")
        react_steps = _phase_steps(
            react_result.trajectory,
            "react",
            start_index=len(cot_steps) + 1,
        )
        prediction = react_result.prediction
        metadata = _hybrid_metadata(
            cot_outcome,
            selected_path="react",
            fallback_used=True,
            extra={"react_termination_reason": react_result.termination_reason},
        )
        return AgentResult(
            prediction=prediction,
            steps=len(cot_steps) + len(react_steps),
            tool_calls=react_result.tool_calls,
            termination_reason=(
                "fallback_react_completed" if prediction else "hybrid_failed"
            ),
            trajectory=cot_steps + react_steps,
            supporting_facts=react_result.supporting_facts,
            metadata=metadata,
        )


def _phase_steps(
    steps: tuple[TrajectoryStep, ...],
    phase: str,
    *,
    start_index: int = 1,
) -> tuple[TrajectoryStep, ...]:
    return tuple(
        replace(step, step_index=start_index + offset, phase=phase)
        for offset, step in enumerate(steps)
    )


def _hybrid_metadata(
    outcome: CoTSCOutcome,
    *,
    selected_path: str,
    fallback_used: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(outcome.result.metadata)
    metadata.update(
        selected_path=selected_path,
        fallback_used=fallback_used,
        cot_sc_paper_threshold=outcome.sample_count / 2,
        cot_sc_threshold_met=outcome.meets_paper_threshold,
    )
    if extra:
        metadata.update(extra)
    return metadata
