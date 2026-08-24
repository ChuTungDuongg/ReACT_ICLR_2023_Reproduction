"""Task evaluators plugged into the shared benchmark runner."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol, Sequence

from react_reproduction.agents.base import AgentResult
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.fever import (
    aggregate_fever_metrics,
    normalize_fever_label,
)
from react_reproduction.evaluation.metrics import (
    aggregate_hotpotqa_metrics,
    answer_metrics,
    joint_metrics,
    supporting_fact_metrics,
)
from react_reproduction.evaluation.schemas import (
    BenchmarkMetrics,
    FeverMetrics,
    FeverPredictionRecord,
    PredictionRecord,
)


Prediction = PredictionRecord | FeverPredictionRecord
Metrics = BenchmarkMetrics | FeverMetrics


class TaskEvaluator(Protocol):
    task: str
    input_name: str
    running_metric_name: str

    def evaluate(
        self,
        example: BenchmarkExample,
        output: AgentResult,
        *,
        method: str,
        latency: float,
    ) -> Prediction: ...

    def aggregate(self, records: Sequence[Prediction], *, runtime: float) -> Metrics: ...

    def log_record(self, logger: logging.Logger, record: Prediction) -> None: ...

    def log_final(self, logger: logging.Logger, metrics: Metrics) -> None: ...


class HotpotQAEvaluator:
    task = "hotpotqa"
    input_name = "Question"
    running_metric_name = "EM"

    def evaluate(
        self,
        example: BenchmarkExample,
        output: AgentResult,
        *,
        method: str,
        latency: float,
    ) -> PredictionRecord:
        answer_em, answer_f1, answer_precision, answer_recall = answer_metrics(
            output.prediction,
            example.gold_answer,
        )
        gold_supporting_facts = _extract_supporting_facts(
            example.metadata.get("supporting_facts")
        )
        sp_em, sp_f1, sp_precision, sp_recall = supporting_fact_metrics(
            output.supporting_facts,
            gold_supporting_facts,
        )
        joint_em, joint_f1, joint_precision, joint_recall = joint_metrics(
            answer_em,
            answer_precision,
            answer_recall,
            sp_em,
            sp_precision,
            sp_recall,
        )
        return PredictionRecord(
            example_id=example.example_id,
            task=self.task,
            method=method,
            question_or_claim=example.input_text,
            gold_answer=example.gold_answer,
            prediction=output.prediction,
            correct=answer_em,
            answer_f1=answer_f1,
            answer_precision=answer_precision,
            answer_recall=answer_recall,
            predicted_supporting_facts=output.supporting_facts,
            gold_supporting_facts=gold_supporting_facts,
            supporting_fact_exact_match=sp_em,
            supporting_fact_f1=sp_f1,
            supporting_fact_precision=sp_precision,
            supporting_fact_recall=sp_recall,
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

    def aggregate(
        self,
        records: Sequence[Prediction],
        *,
        runtime: float,
    ) -> BenchmarkMetrics:
        typed = [record for record in records if isinstance(record, PredictionRecord)]
        if len(typed) != len(records):
            raise TypeError("HotpotQA evaluator received a FEVER prediction record.")
        return aggregate_hotpotqa_metrics(typed, runtime=runtime)

    def log_record(self, logger: logging.Logger, record: Prediction) -> None:
        if not isinstance(record, PredictionRecord):
            raise TypeError("Expected a HotpotQA prediction record.")
        logger.info(
            "Answer | EM: %.4f | F1: %.4f",
            float(record.correct),
            record.answer_f1,
        )
        logger.debug(
            "Answer detail | Precision: %.4f | Recall: %.4f",
            record.answer_precision,
            record.answer_recall,
        )
        logger.debug(
            "Supporting-fact metrics | EM: %.4f | F1: %.4f | Precision: %.4f | "
            "Recall: %.4f | Predicted: %d | Gold: %d",
            record.supporting_fact_exact_match,
            record.supporting_fact_f1,
            record.supporting_fact_precision,
            record.supporting_fact_recall,
            len(record.predicted_supporting_facts),
            len(record.gold_supporting_facts),
        )

    def log_final(self, logger: logging.Logger, metrics: Metrics) -> None:
        if not isinstance(metrics, BenchmarkMetrics):
            raise TypeError("Expected HotpotQA metrics.")
        logger.info("=== FINAL HOTPOTQA METRICS ===")
        for name, value in metrics.official_hotpotqa_metrics().items():
            logger.info("metric.%s=%.6f", name, value)
        logger.info(
            "metric.supporting_fact_prediction_coverage=%.6f",
            metrics.supporting_fact_prediction_coverage,
        )
        _log_operational_metrics(logger, metrics)
        if metrics.supporting_fact_prediction_coverage == 0.0:
            logger.warning(
                "No supporting-fact pairs were predicted. Official sp_* and joint_* "
                "metrics therefore score empty evidence predictions; no evidence was "
                "fabricated."
            )
        logger.info("=== END FINAL HOTPOTQA METRICS ===")


class FeverEvaluator:
    task = "fever"
    input_name = "Claim"
    running_metric_name = "Accuracy"

    def evaluate(
        self,
        example: BenchmarkExample,
        output: AgentResult,
        *,
        method: str,
        latency: float,
    ) -> FeverPredictionRecord:
        prediction = normalize_fever_label(output.prediction)
        invalid = not bool(prediction)
        termination_reason = output.termination_reason
        metadata = dict(output.metadata)
        if invalid:
            metadata.setdefault("raw_prediction", output.prediction)
            metadata.setdefault("parse_error", "No canonical FEVER label was parsed.")
            if output.prediction:
                termination_reason = "invalid_label"
        return FeverPredictionRecord(
            example_id=example.example_id,
            task=self.task,
            method=method,
            claim=example.input_text,
            gold_answer=example.gold_answer,
            prediction=prediction,
            correct=prediction == example.gold_answer,
            invalid=invalid,
            agent_metadata=metadata,
            steps=output.steps,
            tool_calls=output.tool_calls,
            termination_reason=termination_reason,
            latency=latency,
        )

    def aggregate(
        self,
        records: Sequence[Prediction],
        *,
        runtime: float,
    ) -> FeverMetrics:
        typed = [record for record in records if isinstance(record, FeverPredictionRecord)]
        if len(typed) != len(records):
            raise TypeError("FEVER evaluator received a HotpotQA prediction record.")
        return aggregate_fever_metrics(typed, runtime=runtime)

    def log_record(self, logger: logging.Logger, record: Prediction) -> None:
        if not isinstance(record, FeverPredictionRecord):
            raise TypeError("Expected a FEVER prediction record.")
        logger.info("Label accuracy: %.4f", float(record.correct))
        if record.invalid:
            logger.info("Invalid/unparsed FEVER label: true")

    def log_final(self, logger: logging.Logger, metrics: Metrics) -> None:
        if not isinstance(metrics, FeverMetrics):
            raise TypeError("Expected FEVER metrics.")
        logger.info("=== FINAL FEVER METRICS ===")
        logger.info("metric.accuracy=%.6f", metrics.accuracy)
        logger.info("metric.invalid_unparsed=%d", metrics.invalid_unparsed)
        logger.info("metric.per_class_accuracy=%s", dict(metrics.per_class_accuracy))
        logger.info("metric.confusion_matrix=%s", dict(metrics.confusion_matrix))
        _log_operational_metrics(logger, metrics)
        logger.info("=== END FINAL FEVER METRICS ===")


def _extract_supporting_facts(raw_value: Any) -> tuple[tuple[str, int], ...]:
    if raw_value is None:
        return ()
    pairs: list[tuple[str, int]] = []
    if isinstance(raw_value, Mapping):
        values = zip(
            raw_value.get("title", ()),
            raw_value.get("sent_id", ()),
            strict=False,
        )
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        values = (
            item
            for item in raw_value
            if isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
        )
    else:
        return ()
    for title, sentence_id in values:
        cleaned_title = str(title).strip()
        parsed_sentence_id = int(sentence_id)
        if cleaned_title and parsed_sentence_id >= 0:
            pairs.append((cleaned_title, parsed_sentence_id))
    return tuple(pairs)


def _log_operational_metrics(logger: logging.Logger, metrics: Metrics) -> None:
    logger.info("metric.total_examples=%d", metrics.total_examples)
    logger.info("metric.correct=%d", metrics.correct)
    logger.info("metric.incorrect=%d", metrics.incorrect)
    logger.info("metric.average_steps=%.6f", metrics.average_steps)
    logger.info("metric.average_tool_calls=%.6f", metrics.average_tool_calls)
    logger.info("metric.runtime_seconds=%.6f", metrics.runtime)
    logger.info("metric.termination_reasons=%s", dict(metrics.termination_reasons))
