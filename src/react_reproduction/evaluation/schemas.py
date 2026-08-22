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
    steps: int
    tool_calls: int
    termination_reason: str
    latency: float

    def __post_init__(self) -> None:
        if self.steps < 0 or self.tool_calls < 0:
            raise ValueError("steps and tool_calls cannot be negative.")
        if self.latency < 0:
            raise ValueError("latency cannot be negative.")

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
    total_examples: int
    correct: int
    incorrect: int
    average_steps: float
    average_tool_calls: float
    runtime: float
    termination_reasons: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["termination_reasons"] = dict(self.termination_reasons)
        return result
