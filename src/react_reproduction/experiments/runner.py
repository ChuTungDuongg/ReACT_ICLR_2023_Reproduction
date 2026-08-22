"""Model-agnostic HotpotQA benchmark runner with mock inference support."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.metrics import (
    aggregate_hotpotqa_metrics,
    exact_match_score,
)
from react_reproduction.evaluation.schemas import BenchmarkMetrics, PredictionRecord
from react_reproduction.experiments.artifacts import (
    RunArtifacts,
    append_prediction,
    create_run_artifacts,
    write_config,
    write_metrics,
)


@dataclass(frozen=True, slots=True)
class PredictorOutput:
    """Task-neutral output returned by a model or test double."""

    prediction: str
    steps: int = 1
    tool_calls: int = 0
    termination_reason: str = "completed"

    def __post_init__(self) -> None:
        if self.steps < 0 or self.tool_calls < 0:
            raise ValueError("steps and tool_calls cannot be negative.")


class Predictor(Protocol):
    def predict(self, example: BenchmarkExample) -> PredictorOutput: ...


class MockPredictor:
    """Deterministic inference double used only by tests and runner smoke checks."""

    def __init__(
        self,
        predictions_by_id: Mapping[str, str],
        *,
        default_prediction: str = "",
    ) -> None:
        self._predictions_by_id = dict(predictions_by_id)
        self._default_prediction = default_prediction

    def predict(self, example: BenchmarkExample) -> PredictorOutput:
        prediction = self._predictions_by_id.get(
            example.example_id,
            self._default_prediction,
        )
        return PredictorOutput(
            prediction=prediction,
            steps=1,
            tool_calls=0,
            termination_reason="mock_completed",
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    """Complete reproducibility metadata written to ``config.json``."""

    model: str
    dataset: str
    task: str
    method: str
    num_samples: int
    seed: int
    generation: Mapping[str, Any]
    max_agent_steps: int
    timestamp: str = field(default_factory=lambda: _utc_now().isoformat())

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["generation"] = dict(self.generation)
        return result


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """In-memory summary and filesystem locations for a completed run."""

    metrics: BenchmarkMetrics
    predictions: tuple[PredictionRecord, ...]
    artifacts: RunArtifacts


def run_hotpotqa_benchmark(
    examples: Sequence[BenchmarkExample],
    predictor: Predictor,
    config: BenchmarkRunConfig,
    *,
    output_root: Path,
    logger: logging.Logger | None = None,
) -> BenchmarkRunResult:
    """Evaluate a predictor and persist every prediction incrementally."""
    if config.task != "hotpotqa":
        raise ValueError("run_hotpotqa_benchmark requires task='hotpotqa'.")
    if len(examples) != config.num_samples:
        raise ValueError(
            "The number of examples must match config.num_samples: "
            f"{len(examples)} != {config.num_samples}."
        )

    active_logger = logger or logging.getLogger("react_reproduction.experiments")
    artifacts = create_run_artifacts(
        output_root,
        task=config.task,
        method=config.method,
        timestamp=config.timestamp,
    )
    write_config(artifacts, config.to_dict())

    run_started = time.perf_counter()
    prediction_records: list[PredictionRecord] = []
    running_correct = 0

    for index, example in enumerate(examples, start=1):
        example_started = time.perf_counter()
        output = predictor.predict(example)
        latency = time.perf_counter() - example_started
        correct = exact_match_score(output.prediction, example.gold_answer)
        running_correct += int(correct)

        record = PredictionRecord(
            example_id=example.example_id,
            task=config.task,
            method=config.method,
            question_or_claim=example.input_text,
            gold_answer=example.gold_answer,
            prediction=output.prediction,
            correct=correct,
            steps=output.steps,
            tool_calls=output.tool_calls,
            termination_reason=output.termination_reason,
            latency=latency,
        )
        prediction_records.append(record)
        append_prediction(artifacts, record)

        active_logger.info(
            "[%d/%d] example_id=%s correct=%s running_em=%.2f%%",
            index,
            len(examples),
            example.example_id,
            correct,
            100.0 * running_correct / index,
        )

    runtime = time.perf_counter() - run_started
    metrics = aggregate_hotpotqa_metrics(prediction_records, runtime=runtime)
    write_metrics(artifacts, metrics)
    return BenchmarkRunResult(
        metrics=metrics,
        predictions=tuple(prediction_records),
        artifacts=artifacts,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
