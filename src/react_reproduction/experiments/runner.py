"""Model-agnostic HotpotQA benchmark runner with mock inference support."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from react_reproduction import __version__
from react_reproduction.agents.base import AgentResult
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.metrics import (
    aggregate_hotpotqa_metrics,
    answer_metrics,
    joint_metrics,
    supporting_fact_metrics,
)
from react_reproduction.evaluation.schemas import (
    BenchmarkMetrics,
    PredictionRecord,
    TrajectoryRecord,
)
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
    """Deterministic inference double used only by tests and runner smoke checks."""

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
    method_settings: Mapping[str, Any] = field(default_factory=dict)
    device: str = "auto"
    code_version: str = __version__
    timestamp: str = field(default_factory=lambda: _utc_now().isoformat())

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["generation"] = dict(self.generation)
        result["method_settings"] = dict(self.method_settings)
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
    artifacts: RunArtifacts | None = None,
    show_trajectories: bool = False,
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
    active_artifacts = artifacts or create_run_artifacts(
        output_root,
        task=config.task,
        method=config.method,
        timestamp=config.timestamp,
    )
    write_config(active_artifacts, config.to_dict())
    persist_trajectories = config.method in {
        "act",
        "react",
        "react-cot-sc",
        "cot-sc-react",
    }
    if persist_trajectories:
        initialize_trajectories(active_artifacts)

    run_started = time.perf_counter()
    prediction_records: list[PredictionRecord] = []
    running_correct = 0

    for index, example in enumerate(examples, start=1):
        active_logger.info(
            "[%d/%d] example_id=%s",
            index,
            len(examples),
            example.example_id,
        )
        active_logger.info("Question: %s", example.input_text)

        example_started = time.perf_counter()
        output = predictor.predict(example)
        latency = time.perf_counter() - example_started
        answer_em, answer_f1, answer_precision, answer_recall = answer_metrics(
            output.prediction,
            example.gold_answer,
        )
        predicted_supporting_facts = output.supporting_facts
        gold_supporting_facts = _extract_supporting_facts(
            example.metadata.get("supporting_facts")
        )
        (
            supporting_fact_em,
            supporting_fact_f1,
            supporting_fact_precision,
            supporting_fact_recall,
        ) = supporting_fact_metrics(
            predicted_supporting_facts,
            gold_supporting_facts,
        )
        joint_em, joint_f1, joint_precision, joint_recall = joint_metrics(
            answer_em,
            answer_precision,
            answer_recall,
            supporting_fact_em,
            supporting_fact_precision,
            supporting_fact_recall,
        )
        correct = answer_em
        running_correct += int(correct)

        record = PredictionRecord(
            example_id=example.example_id,
            task=config.task,
            method=config.method,
            question_or_claim=example.input_text,
            gold_answer=example.gold_answer,
            prediction=output.prediction,
            correct=correct,
            answer_f1=answer_f1,
            answer_precision=answer_precision,
            answer_recall=answer_recall,
            predicted_supporting_facts=predicted_supporting_facts,
            gold_supporting_facts=gold_supporting_facts,
            supporting_fact_exact_match=supporting_fact_em,
            supporting_fact_f1=supporting_fact_f1,
            supporting_fact_precision=supporting_fact_precision,
            supporting_fact_recall=supporting_fact_recall,
            joint_exact_match=joint_em,
            joint_f1=joint_f1,
            joint_precision=joint_precision,
            joint_recall=joint_recall,
            agent_metadata=output.metadata,
            steps=output.steps,
            tool_calls=output.tool_calls,
            termination_reason=output.termination_reason,
            latency=latency,
        )
        prediction_records.append(record)
        append_prediction(active_artifacts, record)
        if persist_trajectories:
            append_trajectory(
                active_artifacts,
                TrajectoryRecord(
                    example_id=example.example_id,
                    task=config.task,
                    method=config.method,
                    termination_reason=output.termination_reason,
                    steps=output.trajectory,
                ),
            )

        if show_trajectories:
            for step in output.trajectory:
                phase_suffix = f" [{step.phase}]" if step.phase else ""
                active_logger.info("Step %d%s", step.step_index, phase_suffix)
                if step.thought:
                    active_logger.info("Thought: %s", step.thought)
                if step.action:
                    active_logger.info("Action: %s", step.action)
                if step.observation:
                    active_logger.info("Observation: %s", step.observation)
                active_logger.info("Model output: %s", step.model_output)
        displayed_prediction = output.prediction or "<EMPTY>"
        active_logger.info("Prediction: %s", displayed_prediction)
        active_logger.info("Gold: %s", example.gold_answer)
        if output.metadata:
            active_logger.info("Agent metadata: %s", dict(output.metadata))
        active_logger.info(
            "Answer | EM: %.4f | F1: %.4f",
            float(answer_em),
            answer_f1,
        )
        active_logger.debug(
            "Answer detail | Precision: %.4f | Recall: %.4f",
            answer_precision,
            answer_recall,
        )
        active_logger.debug(
            "Supporting-fact metrics | EM: %.4f | F1: %.4f | Precision: %.4f | "
            "Recall: %.4f | Predicted: %d | Gold: %d",
            supporting_fact_em,
            supporting_fact_f1,
            supporting_fact_precision,
            supporting_fact_recall,
            len(predicted_supporting_facts),
            len(gold_supporting_facts),
        )
        active_logger.debug(
            "Joint metrics | EM: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
            joint_em,
            joint_f1,
            joint_precision,
            joint_recall,
        )
        active_logger.info(
            "Result | Correct: %s | Running EM: %.2f%% | Termination: %s | "
            "Steps: %d | Tool calls: %d | Latency: %.3fs",
            correct,
            100.0 * running_correct / index,
            output.termination_reason,
            output.steps,
            output.tool_calls,
            latency,
        )

    runtime = time.perf_counter() - run_started
    metrics = aggregate_hotpotqa_metrics(prediction_records, runtime=runtime)
    write_metrics(active_artifacts, metrics)
    _log_final_metrics(active_logger, metrics)
    return BenchmarkRunResult(
        metrics=metrics,
        predictions=tuple(prediction_records),
        artifacts=active_artifacts,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_supporting_facts(raw_value: Any) -> tuple[tuple[str, int], ...]:
    """Normalize Hugging Face HotpotQA supporting facts into title/id pairs."""
    if raw_value is None:
        return ()
    pairs: list[tuple[str, int]] = []
    if isinstance(raw_value, Mapping):
        titles = raw_value.get("title", ())
        sentence_ids = raw_value.get("sent_id", ())
        for title, sentence_id in zip(titles, sentence_ids, strict=False):
            cleaned_title = str(title).strip()
            parsed_sentence_id = int(sentence_id)
            if cleaned_title and parsed_sentence_id >= 0:
                pairs.append((cleaned_title, parsed_sentence_id))
        return tuple(pairs)
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        for item in raw_value:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
                continue
            if len(item) != 2:
                continue
            cleaned_title = str(item[0]).strip()
            parsed_sentence_id = int(item[1])
            if cleaned_title and parsed_sentence_id >= 0:
                pairs.append((cleaned_title, parsed_sentence_id))
    return tuple(pairs)


def _log_final_metrics(logger: logging.Logger, metrics: BenchmarkMetrics) -> None:
    """Stream every official and operational metric to stdout and run.log."""
    logger.info("=== FINAL HOTPOTQA METRICS ===")
    for name, value in metrics.official_hotpotqa_metrics().items():
        logger.info("metric.%s=%.6f", name, value)
    logger.info(
        "metric.supporting_fact_prediction_coverage=%.6f",
        metrics.supporting_fact_prediction_coverage,
    )
    logger.info("metric.total_examples=%d", metrics.total_examples)
    logger.info("metric.correct=%d", metrics.correct)
    logger.info("metric.incorrect=%d", metrics.incorrect)
    logger.info("metric.average_steps=%.6f", metrics.average_steps)
    logger.info("metric.average_tool_calls=%.6f", metrics.average_tool_calls)
    logger.info("metric.runtime_seconds=%.6f", metrics.runtime)
    logger.info("metric.termination_reasons=%s", dict(metrics.termination_reasons))
    if metrics.supporting_fact_prediction_coverage == 0.0:
        logger.warning(
            "No supporting-fact pairs were predicted. Official sp_* and joint_* "
            "metrics therefore score empty evidence predictions; no evidence was fabricated."
        )
    logger.info("=== END FINAL HOTPOTQA METRICS ===")
