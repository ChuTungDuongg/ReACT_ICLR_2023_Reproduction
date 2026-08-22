"""Unit tests for the official HotpotQA answer, evidence, and joint metrics."""

import pytest

from react_reproduction.evaluation.metrics import (
    answer_metrics,
    exact_match_score,
    joint_metrics,
    normalize_answer,
    supporting_fact_metrics,
)


def test_normalize_answer_removes_articles_punctuation_and_extra_space() -> None:
    assert normalize_answer("  The Eiffel, Tower! ") == "eiffel tower"


def test_exact_match_uses_normalized_answers() -> None:
    assert exact_match_score("An Arthur's Magazine", "arthurs magazine")


def test_exact_match_rejects_different_answers() -> None:
    assert not exact_match_score("Berlin", "Germany")


def test_answer_metrics_match_official_token_overlap_formula() -> None:
    exact_match, f1, precision, recall = answer_metrics("Berlin Germany", "Berlin")
    assert exact_match is False
    assert f1 == pytest.approx(2 / 3)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)


def test_answer_metrics_zero_yes_no_mismatch() -> None:
    assert answer_metrics("yes indeed", "yes") == (False, 0.0, 0.0, 0.0)


def test_supporting_fact_metrics_match_official_set_formula() -> None:
    metrics = supporting_fact_metrics(
        (("Article A", 0), ("Article C", 2)),
        (("Article A", 0), ("Article B", 1)),
    )
    assert metrics == pytest.approx((0.0, 0.5, 0.5, 0.5))


def test_joint_metrics_multiply_answer_and_supporting_fact_scores() -> None:
    metrics = joint_metrics(
        True,
        0.5,
        1.0,
        1.0,
        0.5,
        0.5,
    )
    assert metrics == pytest.approx((1.0, 1 / 3, 0.25, 0.5))
