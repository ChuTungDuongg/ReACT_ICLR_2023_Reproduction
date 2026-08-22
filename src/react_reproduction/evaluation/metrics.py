"""HotpotQA answer normalization and Exact Match evaluation."""

from __future__ import annotations

import re
import string
from collections import Counter
from statistics import fmean
from typing import Sequence

from react_reproduction.evaluation.schemas import BenchmarkMetrics, PredictionRecord


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(answer: str) -> str:
    """Apply the standard HotpotQA/SQuAD normalization procedure."""
    lowered = answer.lower()
    without_punctuation = lowered.translate(_PUNCTUATION_TABLE)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, gold_answer: str) -> bool:
    """Return whether prediction and reference match after normalization."""
    return normalize_answer(prediction) == normalize_answer(gold_answer)


def aggregate_hotpotqa_metrics(
    predictions: Sequence[PredictionRecord],
    *,
    runtime: float,
) -> BenchmarkMetrics:
    """Aggregate prediction records into serializable benchmark metrics."""
    if runtime < 0:
        raise ValueError("runtime cannot be negative.")

    total = len(predictions)
    correct = sum(record.correct for record in predictions)
    termination_reasons = Counter(
        record.termination_reason for record in predictions
    )

    return BenchmarkMetrics(
        exact_match=correct / total if total else 0.0,
        total_examples=total,
        correct=correct,
        incorrect=total - correct,
        average_steps=fmean(record.steps for record in predictions) if total else 0.0,
        average_tool_calls=(
            fmean(record.tool_calls for record in predictions) if total else 0.0
        ),
        runtime=runtime,
        termination_reasons=dict(sorted(termination_reasons.items())),
    )
