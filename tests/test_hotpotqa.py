"""Offline tests for HotpotQA loading and deterministic sampling."""

from __future__ import annotations

from typing import Any

import pytest

from react_reproduction.datasets.hotpotqa import load_hotpotqa


def _records(count: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "id": f"example-{index}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": [f"Article {index}"], "sent_id": [0]},
        }
        for index in range(count)
    ]


def test_sampling_is_deterministic_and_preserves_metadata() -> None:
    records = _records()

    first = load_hotpotqa(4, 42, dataset_loader=lambda *args, **kwargs: records)
    second = load_hotpotqa(4, 42, dataset_loader=lambda *args, **kwargs: records)
    other_seed = load_hotpotqa(4, 7, dataset_loader=lambda *args, **kwargs: records)

    assert [example.example_id for example in first] == [
        example.example_id for example in second
    ]
    assert [example.example_id for example in first] != [
        example.example_id for example in other_seed
    ]
    assert first[0].metadata["question_type"] == "bridge"
    assert first[0].metadata["supporting_facts"] is not None


def test_loader_passes_hugging_face_configuration() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_loader(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        return _records(2)

    load_hotpotqa(
        1,
        42,
        dataset_name="hotpotqa/hotpot_qa",
        subset="distractor",
        split="validation",
        dataset_loader=fake_loader,
    )

    assert calls == [
        (
            ("hotpotqa/hotpot_qa", "distractor"),
            {"split": "validation", "cache_dir": None},
        )
    ]


def test_sampling_rejects_request_larger_than_dataset() -> None:
    with pytest.raises(ValueError, match="dataset contains only 2"):
        load_hotpotqa(3, 42, dataset_loader=lambda *args, **kwargs: _records(2))
