"""Typed project configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv


VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True, slots=True)
class BenchmarkDefaults:
    """Defaults shared by future benchmark commands."""

    num_samples: int = 10
    seed: int = 42
    max_agent_steps: int = 7


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Model generation defaults kept in one configuration location."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("generation.temperature cannot be negative.")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("generation.top_p must be between 0.0 and 1.0.")
        if self.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive.")


@dataclass(frozen=True, slots=True)
class HotpotQAConfig:
    """Hugging Face source used by the HotpotQA benchmark."""

    dataset_name: str = "hotpotqa/hotpot_qa"
    subset: str = "distractor"
    split: str = "validation"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated Sprint 0 configuration."""

    project_name: str
    output_dir: Path
    log_level: str
    hotpotqa: HotpotQAConfig
    benchmark: BenchmarkDefaults
    generation: GenerationConfig


def load_project_config(config_path: Path, *, project_root: Path) -> ProjectConfig:
    """Load YAML configuration and apply documented environment overrides."""
    root = project_root.resolve()
    load_dotenv(root / ".env", override=False)

    resolved_config = _resolve_path(config_path, root)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config}")

    with resolved_config.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    if not isinstance(raw_config, Mapping):
        raise ValueError("The root YAML value must be a mapping.")

    benchmark_data = _mapping(raw_config.get("benchmark"), "benchmark")
    generation_data = _mapping(raw_config.get("generation"), "generation")
    datasets_data = _mapping(raw_config.get("datasets"), "datasets")
    hotpotqa_data = _mapping(datasets_data.get("hotpotqa"), "datasets.hotpotqa")

    project_name = str(raw_config.get("project_name", "react-paper-reproduction"))
    output_value = os.getenv(
        "REACT_OUTPUT_DIR", str(raw_config.get("output_dir", "outputs"))
    )
    log_level = os.getenv(
        "REACT_LOG_LEVEL", str(raw_config.get("log_level", "INFO"))
    ).upper()
    if log_level not in VALID_LOG_LEVELS:
        valid_levels = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ValueError(f"Invalid log level {log_level!r}; expected one of {valid_levels}.")

    hotpotqa = HotpotQAConfig(
        dataset_name=_non_empty_string(
            hotpotqa_data.get("dataset_name", "hotpotqa/hotpot_qa"),
            "datasets.hotpotqa.dataset_name",
        ),
        subset=_non_empty_string(
            hotpotqa_data.get("subset", "distractor"),
            "datasets.hotpotqa.subset",
        ),
        split=_non_empty_string(
            hotpotqa_data.get("split", "validation"),
            "datasets.hotpotqa.split",
        ),
    )
    benchmark = BenchmarkDefaults(
        num_samples=_positive_int(benchmark_data.get("num_samples", 10), "num_samples"),
        seed=int(benchmark_data.get("seed", 42)),
        max_agent_steps=_positive_int(
            benchmark_data.get("max_agent_steps", 7), "max_agent_steps"
        ),
    )
    generation = GenerationConfig(
        temperature=float(generation_data.get("temperature", 0.0)),
        top_p=float(generation_data.get("top_p", 1.0)),
        max_new_tokens=_positive_int(
            generation_data.get("max_new_tokens", 256), "max_new_tokens"
        ),
    )
    return ProjectConfig(
        project_name=project_name,
        output_dir=_resolve_path(Path(output_value), root),
        log_level=log_level,
        hotpotqa=hotpotqa,
        benchmark=benchmark,
        generation=generation,
    )


def _resolve_path(path: Path, project_root: Path) -> Path:
    return (path if path.is_absolute() else project_root / path).resolve()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration field {field_name!r} must be a mapping.")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Configuration field {field_name!r} must be positive.")
    return parsed


def _non_empty_string(value: Any, field_name: str) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise ValueError(f"Configuration field {field_name!r} cannot be empty.")
    return parsed
