"""Tests for structured model-output parsing."""

from react_reproduction.agents.parsing import (
    parse_explicit_final_answer,
    parse_final_answer,
    parse_reasoning,
)


def test_parse_final_answer_prefers_explicit_marker() -> None:
    output = "Reasoning: Several facts.\nFinal Answer: Berlin"
    assert parse_final_answer(output) == "Berlin"


def test_parse_final_answer_falls_back_to_last_non_empty_line() -> None:
    assert parse_final_answer("Some context\nParis\n") == "Paris"


def test_parse_explicit_final_answer_rejects_unlabeled_output() -> None:
    assert parse_explicit_final_answer("Some reasoning\nParis") == ""


def test_parse_reasoning_returns_text_before_final_answer() -> None:
    output = "Reasoning: First fact. Then second fact.\nFinal Answer: Result"
    assert parse_reasoning(output) == "First fact. Then second fact."


def test_parse_reasoning_recovers_completion_after_prompt_thought_prefix() -> None:
    output = "Evidence supports the claim.\nAnswer: SUPPORTS"
    assert parse_reasoning(output) == "Evidence supports the claim."
