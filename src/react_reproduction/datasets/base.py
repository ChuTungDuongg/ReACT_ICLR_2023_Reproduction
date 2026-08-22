"""Dataset-neutral input schema shared by benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One normalized example consumed by an experiment runner."""

    example_id: str
    input_text: str
    gold_answer: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id cannot be empty.")
        if not self.input_text.strip():
            raise ValueError("input_text cannot be empty.")
        if not self.gold_answer.strip():
            raise ValueError("gold_answer cannot be empty.")
