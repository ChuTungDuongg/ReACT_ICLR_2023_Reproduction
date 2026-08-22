"""HotpotQA loading and deterministic subset sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from react_reproduction.datasets.base import BenchmarkExample


DEFAULT_DATASET_NAME = "hotpotqa/hotpot_qa"
DEFAULT_SUBSET = "distractor"
DEFAULT_SPLIT = "validation"


class IndexableRecords(Protocol):
    """Minimal interface implemented by Hugging Face Dataset and test doubles."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


DatasetLoader = Callable[..., IndexableRecords]


def load_hotpotqa(
    num_samples: int,
    seed: int,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subset: str = DEFAULT_SUBSET,
    split: str = DEFAULT_SPLIT,
    cache_dir: Path | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> list[BenchmarkExample]:
    """Load and deterministically sample HotpotQA from Hugging Face.

    ``dataset_loader`` is injectable so tests never need network access.
    """
    _validate_positive_sample_count(num_samples)
    loader = dataset_loader or _huggingface_load_dataset
    records = loader(
        dataset_name,
        subset,
        split=split,
        cache_dir=str(cache_dir.resolve()) if cache_dir is not None else None,
    )
    return sample_hotpotqa_records(records, num_samples=num_samples, seed=seed)


def sample_hotpotqa_records(
    records: IndexableRecords,
    *,
    num_samples: int,
    seed: int,
) -> list[BenchmarkExample]:
    """Select a reproducible subset without mutating the source dataset."""
    _validate_positive_sample_count(num_samples)
    dataset_size = len(records)
    if num_samples > dataset_size:
        raise ValueError(
            f"Requested {num_samples} samples, but the dataset contains "
            f"only {dataset_size}."
        )

    sampled_indices = random.Random(seed).sample(range(dataset_size), num_samples)
    return [_record_to_example(records[index]) for index in sampled_indices]


def _record_to_example(record: Mapping[str, Any]) -> BenchmarkExample:
    required_fields = ("id", "question", "answer")
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"HotpotQA record is missing required field(s): {missing}.")

    return BenchmarkExample(
        example_id=str(record["id"]).strip(),
        input_text=str(record["question"]).strip(),
        gold_answer=str(record["answer"]).strip(),
        metadata={
            "question_type": record.get("type"),
            "level": record.get("level"),
            "supporting_facts": record.get("supporting_facts"),
        },
    )


def _huggingface_load_dataset(*args: Any, **kwargs: Any) -> IndexableRecords:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'datasets' package is required to load HotpotQA. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from error
    return load_dataset(*args, **kwargs)


def _validate_positive_sample_count(num_samples: int) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
