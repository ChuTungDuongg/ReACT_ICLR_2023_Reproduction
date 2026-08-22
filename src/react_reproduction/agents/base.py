"""Shared agent contracts and trajectory primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from react_reproduction.datasets.base import BenchmarkExample


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One inspectable step produced while answering an example."""

    step_index: int
    model_output: str
    thought: str | None = None
    action: str | None = None
    observation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Task-neutral result returned to the experiment runner."""

    prediction: str
    steps: int
    tool_calls: int
    termination_reason: str
    trajectory: tuple[TrajectoryStep, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.steps < 0 or self.tool_calls < 0:
            raise ValueError("steps and tool_calls cannot be negative.")


class BaseAgent(ABC):
    """Common interface implemented by every prompting method."""

    @abstractmethod
    def predict(self, example: BenchmarkExample) -> AgentResult:
        """Answer one normalized benchmark example."""
