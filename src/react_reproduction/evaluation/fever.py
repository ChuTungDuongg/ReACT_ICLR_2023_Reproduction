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
_SURFACE_PATTERN = re.compile(
    r"^(?:"
    + "|".join(
        re.escape(surface)
        for surface in sorted(_SURFACE_FORMS, key=len, reverse=True)
    )
    + r")(?=$|[\s.,;:!?(])",
    flags=re.IGNORECASE,
)


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
    answer_payloads = [
        match.group(1)
        for line in non_empty
        if (match := _ANSWER_PREFIX.match(line)) is not None
    ]
    if answer_payloads:
        parsed = [_parse_fever_answer_payload(payload) for payload in answer_payloads]
        if all(parsed) and len(set(parsed)) == 1:
            return parsed[0]
        return ""

    first_line_label = normalize_fever_label(non_empty[0])
    if not first_line_label:
        return ""
    for line in non_empty[1:]:
        repeated_label = normalize_fever_label(line)
        if repeated_label and repeated_label != first_line_label:
            return ""
        if re.match(r"^(?:or|alternatively|/)\s+", line, flags=re.IGNORECASE):
            if _parse_fever_answer_payload(
                re.sub(
                    r"^(?:or|alternatively|/)\s+",
                    "",
                    line,
                    flags=re.IGNORECASE,
                )
            ):
                return ""
    return first_line_label


def _parse_fever_answer_payload(payload: str) -> str:
    """Accept a leading FEVER label with only punctuation or parenthetical prose."""
    stripped = payload.strip()
    match = _SURFACE_PATTERN.match(stripped)
    if match is None:
        return ""
    label = normalize_fever_label(match.group(0))
    suffix = stripped[match.end() :].strip()
    if not suffix or not suffix.strip(".!,;:"):
        return label
    if suffix.startswith("(") and suffix.endswith(")"):
        inner = suffix[1:-1].strip()
        if inner and not re.search(
            r"(?:\bor\b|/)\s*(?:supports|refutes|not\s+enough\s+info(?:rmation)?)\b",
            inner,
            flags=re.IGNORECASE,
        ):
            return label
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
