"""Task-neutral benchmark runner with shared batching and artifact persistence."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from react_reproduction import __version__
from react_reproduction.agents.base import AgentResult
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.evaluators import (
    FeverEvaluator,
    HotpotQAEvaluator,
    Metrics,
    Prediction,
    TaskEvaluator,
)
from react_reproduction.evaluation.schemas import TrajectoryRecord
from react_reproduction.experiments.artifacts import (
    RunArtifacts,
    append_prediction,
    append_trajectory,
    create_run_artifacts,
    initialize_trajectories,
    write_config,
    write_metrics,
)


class Predictor(Protocol):
    def predict(self, example: BenchmarkExample) -> AgentResult: ...


class MockPredictor:
    """Deterministic inference double used by tests and smoke checks."""

    def __init__(
        self,
        predictions_by_id: Mapping[str, str],
        *,
        default_prediction: str = "",
    ) -> None:
        self._predictions_by_id = dict(predictions_by_id)
        self._default_prediction = default_prediction

    def predict(self, example: BenchmarkExample) -> AgentResult:
        prediction = self._predictions_by_id.get(
            example.example_id,
            self._default_prediction,
        )
        return AgentResult(
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
    batch_size: int = 1
    method_settings: Mapping[str, Any] = field(default_factory=dict)
    device: str = "auto"
    dataset_split: str | None = None
    dataset_subset: str | None = None
    dataset_revision: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    sampling_metadata: Mapping[str, Any] = field(default_factory=dict)
    code_version: str = __version__
    timestamp: str = field(default_factory=lambda: _utc_now().isoformat())

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["generation"] = dict(self.generation)
        result["method_settings"] = dict(self.method_settings)
        result["sampling_metadata"] = dict(self.sampling_metadata)
        result["split"] = self.dataset_split
        result["subset"] = self.dataset_subset
        result.update(
            {
                key: self.generation.get(key)
                for key in ("temperature", "top_p", "max_new_tokens")
            }
        )
        for key in (
            "cot_sc_samples",
            "cot_sc_temperature",
            "cot_sc_threshold",
        ):
            result[key] = self.method_settings.get(key)
        return result


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    metrics: Metrics
    predictions: tuple[Prediction, ...]
    artifacts: RunArtifacts


def run_benchmark(
    examples: Sequence[BenchmarkExample],
    predictor: Predictor,
    config: BenchmarkRunConfig,
    *,
    evaluator: TaskEvaluator,
    output_root: Path,
    logger: logging.Logger | None = None,
    artifacts: RunArtifacts | None = None,
    show_trajectories: bool = False,
) -> BenchmarkRunResult:
    """Evaluate one task through the shared batching and persistence path."""
    if config.task != evaluator.task:
        raise ValueError(
            f"Evaluator task {evaluator.task!r} does not match {config.task!r}."
        )
    if len(examples) != config.num_samples:
        raise ValueError(
            "The number of examples must match config.num_samples: "
            f"{len(examples)} != {config.num_samples}."
        )

    active_logger = logger or logging.getLogger("react_reproduction.experiments")
    active_artifacts = artifacts or create_run_artifacts(
        output_root,
        task=config.task,
        method=config.method,
        timestamp=config.timestamp,
    )
    config_payload = config.to_dict()
    config_payload["sample_ids"] = [example.example_id for example in examples]
    write_config(active_artifacts, config_payload)
    persist_trajectories = config.method in {
        "cot-sc",
        "act",
        "react",
        "react-cot-sc",
        "cot-sc-react",
    }
    if persist_trajectories:
        initialize_trajectories(active_artifacts)

    run_started = time.perf_counter()
    records: list[Prediction] = []
    running_correct = 0

    for index, example, output, latency in _predict_in_batches(
        examples,
        predictor,
        batch_size=config.batch_size,
        logger=active_logger,
        input_name=evaluator.input_name,
    ):
        record = evaluator.evaluate(
            example,
            output,
            method=config.method,
            latency=latency,
        )
        records.append(record)
        running_correct += int(record.correct)
        append_prediction(active_artifacts, record)
        if persist_trajectories:
            append_trajectory(
                active_artifacts,
                TrajectoryRecord(
                    example_id=example.example_id,
                    task=config.task,
                    method=config.method,
                    termination_reason=record.termination_reason,
                    steps=output.trajectory,
                ),
            )

        if show_trajectories:
            _log_trajectory(active_logger, output)
        active_logger.info("Prediction: %s", record.prediction or "<EMPTY>")
        active_logger.info("Gold: %s", example.gold_answer)
        _log_agent_metadata(active_logger, output.metadata)
        evaluator.log_record(active_logger, record)
        active_logger.info(
            "Result | Correct: %s | Running %s: %.2f%% | Termination: %s | "
            "Steps: %d | Tool calls: %d | Latency: %.3fs",
            record.correct,
            evaluator.running_metric_name,
            100.0 * running_correct / index,
            record.termination_reason,
            record.steps,
            record.tool_calls,
            latency,
        )

    runtime = time.perf_counter() - run_started
    metrics = evaluator.aggregate(records, runtime=runtime)
    write_metrics(active_artifacts, metrics)
    evaluator.log_final(active_logger, metrics)
    return BenchmarkRunResult(
        metrics=metrics,
        predictions=tuple(records),
        artifacts=active_artifacts,
    )


def run_hotpotqa_benchmark(
    examples: Sequence[BenchmarkExample],
    predictor: Predictor,
    config: BenchmarkRunConfig,
    *,
    output_root: Path,
    logger: logging.Logger | None = None,
    artifacts: RunArtifacts | None = None,
    show_trajectories: bool = False,
) -> BenchmarkRunResult:
    return run_benchmark(
        examples,
        predictor,
        config,
        evaluator=HotpotQAEvaluator(),
        output_root=output_root,
        logger=logger,
        artifacts=artifacts,
        show_trajectories=show_trajectories,
    )


def run_fever_benchmark(
    examples: Sequence[BenchmarkExample],
    predictor: Predictor,
    config: BenchmarkRunConfig,
    *,
    output_root: Path,
    logger: logging.Logger | None = None,
    artifacts: RunArtifacts | None = None,
    show_trajectories: bool = False,
) -> BenchmarkRunResult:
    return run_benchmark(
        examples,
        predictor,
        config,
        evaluator=FeverEvaluator(),
        output_root=output_root,
        logger=logger,
        artifacts=artifacts,
        show_trajectories=show_trajectories,
    )


def _predict_in_batches(
    examples: Sequence[BenchmarkExample],
    predictor: Predictor,
    *,
    batch_size: int,
    logger: logging.Logger,
    input_name: str = "Question",
) -> Iterator[tuple[int, BenchmarkExample, AgentResult, float]]:
    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch)
        logger.info(
            "Inference batch [%d-%d/%d] size=%d",
            batch_start + 1,
            batch_end,
            len(examples),
            len(batch),
        )
        for offset, example in enumerate(batch, start=batch_start + 1):
            logger.info("[%d/%d] example_id=%s", offset, len(examples), example.example_id)
            logger.info("%s: %s", input_name, example.input_text)

        batch_started = time.perf_counter()
        batch_predict = getattr(predictor, "predict_batch", None)
        if len(batch) > 1 and callable(batch_predict):
            outputs = tuple(batch_predict(batch))
        else:
            outputs = tuple(predictor.predict(example) for example in batch)
        batch_latency = time.perf_counter() - batch_started
        if len(outputs) != len(batch):
            raise RuntimeError(
                "Predictor batch output count does not match input count: "
                f"{len(outputs)} != {len(batch)}."
            )
        amortized_latency = batch_latency / len(batch)
        for index, example, output in zip(
            range(batch_start + 1, batch_end + 1),
            batch,
            outputs,
            strict=True,
        ):
            if not isinstance(output, AgentResult):
                raise TypeError("Predictor outputs must be AgentResult instances.")
            yield index, example, output, amortized_latency


def _log_trajectory(logger: logging.Logger, output: AgentResult) -> None:
    for step in output.trajectory:
        phase_suffix = f" [{step.phase}]" if step.phase else ""
        logger.info("Step %d%s", step.step_index, phase_suffix)
        if step.thought:
            logger.info("Thought: %s", step.thought)
        if step.action:
            logger.info("Action: %s", step.action)
        if step.observation:
            logger.info("Observation: %s", step.observation)
        logger.info("Model output: %s", step.model_output)


def _log_agent_metadata(logger: logging.Logger, metadata: Mapping[str, Any]) -> None:
    if not metadata:
        return
    if "winning_count" in metadata:
        logger.info("Winning label: %s", metadata.get("winning_label") or "<EMPTY>")
        logger.info(
            "Winning count: %d/%d",
            metadata["winning_count"],
            metadata.get("cot_sc_samples", 0),
        )
        logger.info("Votes: %s", metadata.get("vote_distribution", {}))
        if "execution_path" in metadata:
            logger.info("Execution path: %s", metadata["execution_path"])
        return
    logger.info("Agent metadata: %s", dict(metadata))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
