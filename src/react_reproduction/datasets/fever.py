"""FEVER claim-only loading and deterministic subset sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.fever import normalize_fever_label


DEFAULT_DATASET_NAME = "ysymyth/ReAct"
DEFAULT_SUBSET = "paper_dev"
DEFAULT_SPLIT = "dev"
DEFAULT_SOURCE_REVISION = "6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9"


class IndexableRecords(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


DatasetLoader = Callable[..., IndexableRecords]


def load_fever(
    num_samples: int,
    seed: int,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    subset: str = DEFAULT_SUBSET,
    split: str = DEFAULT_SPLIT,
    source_revision: str = DEFAULT_SOURCE_REVISION,
    cache_dir: Path | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> list[BenchmarkExample]:
    """Load FEVER labels and claims without exposing gold evidence."""
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    loader = dataset_loader or _huggingface_load_dataset
    source_url = (
        f"https://raw.githubusercontent.com/{dataset_name}/"
        f"{source_revision}/data/{subset}.jsonl"
    )
    records = loader(
        "json",
        data_files={split: source_url},
        split=split,
        cache_dir=str(cache_dir.resolve()) if cache_dir is not None else None,
    )
    return sample_fever_records(records, num_samples=num_samples, seed=seed)


def sample_fever_records(
    records: IndexableRecords,
    *,
    num_samples: int,
    seed: int,
) -> list[BenchmarkExample]:
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if num_samples > len(records):
        raise ValueError(
            f"Requested {num_samples} samples, but the dataset contains only "
            f"{len(records)}."
        )
    indices = random.Random(seed).sample(range(len(records)), num_samples)
    return [_record_to_example(records[index]) for index in indices]


def _record_to_example(record: Mapping[str, Any]) -> BenchmarkExample:
    missing = [field for field in ("id", "claim", "label") if field not in record]
    if missing:
        raise ValueError(f"FEVER record is missing required field(s): {', '.join(missing)}.")
    label = normalize_fever_label(str(record["label"]))
    if not label:
        raise ValueError(f"FEVER record has an invalid gold label: {record['label']!r}.")
    return BenchmarkExample(
        example_id=str(record["id"]).strip(),
        input_text=str(record["claim"]).strip(),
        gold_answer=label,
        metadata={"source_index": record.get("id")},
    )


def _huggingface_load_dataset(*args: Any, **kwargs: Any) -> IndexableRecords:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'datasets' package is required to load FEVER. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from error
    return load_dataset(*args, **kwargs)
