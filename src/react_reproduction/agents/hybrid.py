"""Paper-style fallback policies combining ReAct and CoT-SC."""

from __future__ import annotations

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
        react_result = self._react.predict(example)
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

        cot_outcome = self._cot_sc.predict_with_consensus(example)
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
        cot_outcome = self._cot_sc.predict_with_consensus(example)
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

        react_result = self._react.predict(example)
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
