"""Creation and serialization of isolated experiment run artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from react_reproduction.evaluation.schemas import (
    BenchmarkMetrics,
    PredictionRecord,
    TrajectoryRecord,
)


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Resolved paths belonging to exactly one benchmark run."""

    run_directory: Path
    config_path: Path
    metrics_path: Path
    predictions_path: Path
    trajectories_path: Path
    run_log_path: Path


def create_run_artifacts(
    output_root: Path,
    *,
    task: str,
    method: str,
    timestamp: str,
) -> RunArtifacts:
    """Create a collision-safe ``task/method/timestamp`` output directory."""
    safe_task = _validate_path_component(task, "task")
    safe_method = _validate_path_component(method, "method")
    run_name = _timestamp_to_run_name(timestamp)
    parent = output_root.resolve() / safe_task / safe_method
    parent.mkdir(parents=True, exist_ok=True)

    run_directory = _create_unique_directory(parent, run_name)
    predictions_path = run_directory / "predictions.jsonl"
    predictions_path.touch(exist_ok=False)

    return RunArtifacts(
        run_directory=run_directory,
        config_path=run_directory / "config.json",
        metrics_path=run_directory / "metrics.json",
        predictions_path=predictions_path,
        trajectories_path=run_directory / "trajectories.jsonl",
        run_log_path=run_directory / "run.log",
    )


def write_config(artifacts: RunArtifacts, config: Mapping[str, Any]) -> None:
    _write_json(artifacts.config_path, config)


def append_prediction(artifacts: RunArtifacts, record: PredictionRecord) -> None:
    with artifacts.predictions_path.open("a", encoding="utf-8", newline="\n") as file:
        json.dump(record.to_dict(), file, ensure_ascii=False)
        file.write("\n")
        file.flush()


def initialize_trajectories(artifacts: RunArtifacts) -> None:
    """Create the trajectory stream before an interactive benchmark begins."""
    artifacts.trajectories_path.touch(exist_ok=False)


def append_trajectory(artifacts: RunArtifacts, record: TrajectoryRecord) -> None:
    with artifacts.trajectories_path.open("a", encoding="utf-8", newline="\n") as file:
        json.dump(record.to_dict(), file, ensure_ascii=False)
        file.write("\n")
        file.flush()


def write_metrics(artifacts: RunArtifacts, metrics: BenchmarkMetrics) -> None:
    _write_json(artifacts.metrics_path, metrics.to_dict())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
    temporary_path.replace(path)


def _timestamp_to_run_name(timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid ISO-8601 timestamp: {timestamp!r}.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _create_unique_directory(parent: Path, run_name: str) -> Path:
    for suffix in range(1_000):
        candidate_name = run_name if suffix == 0 else f"{run_name}_{suffix:02d}"
        candidate = parent / candidate_name
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a unique run directory under {parent}.")


def _validate_path_component(value: str, field_name: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {field_name} path component: {value!r}.")
    return value
