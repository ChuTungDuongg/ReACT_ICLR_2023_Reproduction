"""Conservative FEVER label parsing and Accuracy evaluation."""

from __future__ import annotations

import re
from collections import Counter
from statistics import fmean
from typing import Sequence

from react_reproduction.evaluation.schemas import FeverMetrics, FeverPredictionRecord


FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
_ANSWER_PREFIX = re.compile(
    r"^(?:final\s+answer|answer)\s*:\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)
_SURFACE_FORMS = {
    "support": "SUPPORTS",
    "supports": "SUPPORTS",
    "supported": "SUPPORTS",
    "claim is supported": "SUPPORTS",
    "the claim is supported": "SUPPORTS",
    "refute": "REFUTES",
    "refutes": "REFUTES",
    "refuted": "REFUTES",
    "claim is refuted": "REFUTES",
    "the claim is refuted": "REFUTES",
    "claim is false": "REFUTES",
    "the claim is false": "REFUTES",
    "not enough info": "NOT ENOUGH INFO",
    "not enough information": "NOT ENOUGH INFO",
    "insufficient information": "NOT ENOUGH INFO",
}


def normalize_fever_label(value: str) -> str:
    """Map only unambiguous FEVER surface forms to a canonical label."""
    normalized = " ".join(value.strip().casefold().split())
    normalized = normalized.strip(" \t\r\n.!,;:\"'")
    return _SURFACE_FORMS.get(normalized, "")


def parse_fever_label(model_output: str) -> str:
    """Parse an explicit final answer or a single bare label conservatively."""
    non_empty = [line.strip() for line in model_output.splitlines() if line.strip()]
    if not non_empty:
        return ""
    for line in reversed(non_empty):
        match = _ANSWER_PREFIX.match(line)
        if match is not None:
            return normalize_fever_label(match.group(1))
    if len(non_empty) == 1:
        return normalize_fever_label(non_empty[0])
    return ""


def aggregate_fever_metrics(
    predictions: Sequence[FeverPredictionRecord],
    *,
    runtime: float,
) -> FeverMetrics:
    if runtime < 0:
        raise ValueError("runtime cannot be negative.")
    total = len(predictions)
    correct = sum(record.correct for record in predictions)
    gold_counts = Counter(record.gold_answer for record in predictions)
    predicted_counts = Counter(
        record.prediction if record.prediction else "INVALID" for record in predictions
    )
    confusion: dict[str, dict[str, int]] = {
        label: {predicted: 0 for predicted in (*FEVER_LABELS, "INVALID")}
        for label in FEVER_LABELS
    }
    for record in predictions:
        predicted = record.prediction if record.prediction else "INVALID"
        confusion[record.gold_answer][predicted] += 1
    per_class_accuracy = {
        label: (
            confusion[label][label] / gold_counts[label]
            if gold_counts[label]
            else 0.0
        )
        for label in FEVER_LABELS
    }
    return FeverMetrics(
        accuracy=correct / total if total else 0.0,
        total_examples=total,
        correct=correct,
        incorrect=total - correct,
        invalid_unparsed=sum(record.invalid for record in predictions),
        average_steps=fmean(record.steps for record in predictions) if total else 0.0,
        average_tool_calls=(
            fmean(record.tool_calls for record in predictions) if total else 0.0
        ),
        runtime=runtime,
        termination_reasons=dict(
            sorted(Counter(record.termination_reason for record in predictions).items())
        ),
        gold_label_distribution=dict(sorted(gold_counts.items())),
        predicted_label_distribution=dict(sorted(predicted_counts.items())),
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion,
    )
