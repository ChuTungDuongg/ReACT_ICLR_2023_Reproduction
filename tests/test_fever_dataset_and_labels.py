"""FEVER claim-only dataset and conservative label parsing tests."""

from __future__ import annotations

import pytest

from react_reproduction.datasets.fever import load_fever, sample_fever_records
from react_reproduction.evaluation.fever import (
    normalize_fever_label,
    parse_fever_label,
)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("SUPPORTS", "SUPPORTS"),
        ("support", "SUPPORTS"),
        ("The claim is supported.", "SUPPORTS"),
        ("REFUTES", "REFUTES"),
        ("refuted", "REFUTES"),
        ("The claim is false.", "REFUTES"),
        ("NOT ENOUGH INFO", "NOT ENOUGH INFO"),
        ("not enough information", "NOT ENOUGH INFO"),
        ("insufficient information", "NOT ENOUGH INFO"),
    ],
)
def test_normalize_fever_label_accepts_safe_surface_forms(
    surface: str,
    expected: str,
) -> None:
    assert normalize_fever_label(surface) == expected


@pytest.mark.parametrize(
    "ambiguous",
    [
        "probably supports",
        "I do not know",
        "supports or refutes",
        "The claim may be false",
        "",
    ],
)
def test_normalize_fever_label_rejects_ambiguous_outputs(ambiguous: str) -> None:
    assert normalize_fever_label(ambiguous) == ""


def test_parse_fever_label_requires_explicit_answer_for_multiline_output() -> None:
    assert parse_fever_label("Thought: evidence agrees.\nAnswer: supports") == "SUPPORTS"
    assert parse_fever_label("Thought: maybe supports.\nNo final label") == ""


def test_fever_sampling_is_deterministic_and_claim_only() -> None:
    records = [
        {
            "id": index,
            "claim": f"Claim {index}",
            "label": ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")[index % 3],
            "evidence": [["SECRET_GOLD_EVIDENCE", index]],
        }
        for index in range(10)
    ]

    first = sample_fever_records(records, num_samples=5, seed=42)
    second = sample_fever_records(records, num_samples=5, seed=42)

    assert [item.example_id for item in first] == [item.example_id for item in second]
    assert all("evidence" not in item.metadata for item in first)
    assert all("SECRET_GOLD_EVIDENCE" not in item.input_text for item in first)


def test_fever_loader_passes_the_configured_hugging_face_source() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def loader(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls.append((args, kwargs))
        return [{"id": "1", "claim": "A claim.", "label": "SUPPORTS"}]

    examples = load_fever(
        1,
        7,
        dataset_name="ysymyth/ReAct",
        subset="paper_dev",
        split="dev",
        source_revision="pinned-revision",
        dataset_loader=loader,
    )

    assert examples[0].input_text == "A claim."
    assert calls[0][0] == ("json",)
    assert calls[0][1]["split"] == "dev"
    assert calls[0][1]["data_files"] == {
        "dev": (
            "https://raw.githubusercontent.com/ysymyth/ReAct/"
            "pinned-revision/data/paper_dev.jsonl"
        )
    }


def test_fever_loader_rejects_invalid_gold_label() -> None:
    with pytest.raises(ValueError, match="invalid gold label"):
        sample_fever_records(
            [{"id": "1", "claim": "A claim.", "label": "MAYBE"}],
            num_samples=1,
            seed=42,
        )
