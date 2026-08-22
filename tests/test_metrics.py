"""Unit tests for HotpotQA normalization and Exact Match."""

from react_reproduction.evaluation.metrics import exact_match_score, normalize_answer


def test_normalize_answer_removes_articles_punctuation_and_extra_space() -> None:
    assert normalize_answer("  The Eiffel, Tower! ") == "eiffel tower"


def test_exact_match_uses_normalized_answers() -> None:
    assert exact_match_score("An Arthur's Magazine", "arthurs magazine")


def test_exact_match_rejects_different_answers() -> None:
    assert not exact_match_score("Berlin", "Germany")
