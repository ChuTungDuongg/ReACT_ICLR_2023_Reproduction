"""Serializable schemas for predictions and aggregate metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from react_reproduction.agents.base import TrajectoryStep


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One model prediction with metadata required for later analysis."""

    example_id: str
    task: str
    method: str
    question_or_claim: str
    gold_answer: str
    prediction: str
    correct: bool
    answer_f1: float
    answer_precision: float
    answer_recall: float
    predicted_supporting_facts: tuple[tuple[str, int], ...]
    gold_supporting_facts: tuple[tuple[str, int], ...]
    supporting_fact_exact_match: float
    supporting_fact_f1: float
    supporting_fact_precision: float
    supporting_fact_recall: float
    joint_exact_match: float
    joint_f1: float
    joint_precision: float
    joint_recall: float
    steps: int
    tool_calls: int
    termination_reason: str
    latency: float

    def __post_init__(self) -> None:
        if self.steps < 0 or self.tool_calls < 0:
            raise ValueError("steps and tool_calls cannot be negative.")
        if self.latency < 0:
            raise ValueError("latency cannot be negative.")
        metric_values = (
            self.answer_f1,
            self.answer_precision,
            self.answer_recall,
            self.supporting_fact_exact_match,
            self.supporting_fact_f1,
            self.supporting_fact_precision,
            self.supporting_fact_recall,
            self.joint_exact_match,
            self.joint_f1,
            self.joint_precision,
            self.joint_recall,
        )
        if any(not 0.0 <= value <= 1.0 for value in metric_values):
            raise ValueError("HotpotQA metric values must be between 0.0 and 1.0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """One complete agent trajectory serialized independently per example."""

    example_id: str
    task: str
    method: str
    termination_reason: str
    steps: tuple[TrajectoryStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task": self.task,
            "method": self.method,
            "termination_reason": self.termination_reason,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Aggregate HotpotQA benchmark metrics stored in ``metrics.json``."""

    exact_match: float
    f1: float
    precision: float
    recall: float
    supporting_fact_exact_match: float
    supporting_fact_f1: float
    supporting_fact_precision: float
    supporting_fact_recall: float
    joint_exact_match: float
    joint_f1: float
    joint_precision: float
    joint_recall: float
    supporting_fact_prediction_coverage: float
    total_examples: int
    correct: int
    incorrect: int
    average_steps: float
    average_tool_calls: float
    runtime: float
    termination_reasons: Mapping[str, int] = field(default_factory=dict)

    def official_hotpotqa_metrics(self) -> dict[str, float]:
        """Return the names used by the official HotpotQA evaluator."""
        return {
            "em": self.exact_match,
            "f1": self.f1,
            "prec": self.precision,
            "recall": self.recall,
            "sp_em": self.supporting_fact_exact_match,
            "sp_f1": self.supporting_fact_f1,
            "sp_prec": self.supporting_fact_precision,
            "sp_recall": self.supporting_fact_recall,
            "joint_em": self.joint_exact_match,
            "joint_f1": self.joint_f1,
            "joint_prec": self.joint_precision,
            "joint_recall": self.joint_recall,
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["termination_reasons"] = dict(self.termination_reasons)
        result["official_hotpotqa"] = self.official_hotpotqa_metrics()
        return result
