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


def answer_metrics(
    prediction: str,
    gold_answer: str,
) -> tuple[bool, float, float, float]:
    """Return official HotpotQA answer EM, F1, precision, and recall."""
    exact_match = exact_match_score(prediction, gold_answer)
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold_answer)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers
        or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return exact_match, 0.0, 0.0, 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    matching_tokens = sum(common.values())
    if matching_tokens == 0:
        return exact_match, 0.0, 0.0, 0.0

    precision = matching_tokens / len(prediction_tokens)
    recall = matching_tokens / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return exact_match, f1, precision, recall


def supporting_fact_metrics(
    prediction: Sequence[tuple[str, int]],
    gold: Sequence[tuple[str, int]],
) -> tuple[float, float, float, float]:
    """Return official supporting-fact EM, F1, precision, and recall."""
    predicted_set = set(prediction)
    gold_set = set(gold)
    true_positives = len(predicted_set & gold_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    exact_match = 1.0 if false_positives + false_negatives == 0 else 0.0
    return exact_match, f1, precision, recall


def joint_metrics(
    answer_exact_match: bool,
    answer_precision: float,
    answer_recall: float,
    supporting_fact_exact_match: float,
    supporting_fact_precision: float,
    supporting_fact_recall: float,
) -> tuple[float, float, float, float]:
    """Return official joint EM, F1, precision, and recall."""
    precision = answer_precision * supporting_fact_precision
    recall = answer_recall * supporting_fact_recall
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = float(answer_exact_match) * supporting_fact_exact_match
    return exact_match, f1, precision, recall


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
        f1=fmean(record.answer_f1 for record in predictions) if total else 0.0,
        precision=(
            fmean(record.answer_precision for record in predictions) if total else 0.0
        ),
        recall=(
            fmean(record.answer_recall for record in predictions) if total else 0.0
        ),
        supporting_fact_exact_match=(
            fmean(record.supporting_fact_exact_match for record in predictions)
            if total
            else 0.0
        ),
        supporting_fact_f1=(
            fmean(record.supporting_fact_f1 for record in predictions)
            if total
            else 0.0
        ),
        supporting_fact_precision=(
            fmean(record.supporting_fact_precision for record in predictions)
            if total
            else 0.0
        ),
        supporting_fact_recall=(
            fmean(record.supporting_fact_recall for record in predictions)
            if total
            else 0.0
        ),
        joint_exact_match=(
            fmean(record.joint_exact_match for record in predictions)
            if total
            else 0.0
        ),
        joint_f1=(
            fmean(record.joint_f1 for record in predictions) if total else 0.0
        ),
        joint_precision=(
            fmean(record.joint_precision for record in predictions)
            if total
            else 0.0
        ),
        joint_recall=(
            fmean(record.joint_recall for record in predictions)
            if total
            else 0.0
        ),
        supporting_fact_prediction_coverage=(
            sum(bool(record.predicted_supporting_facts) for record in predictions)
            / total
            if total
            else 0.0
        ),
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
