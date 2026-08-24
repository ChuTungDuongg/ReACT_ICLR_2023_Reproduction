"""Network- and model-free smoke tests for the repository bootstrap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from react_reproduction.cli import DEFAULT_MODEL, METHODS, _resolve_task_runtime, build_parser
from react_reproduction.config import load_project_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_main_help() -> None:
    completed = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Reproduce and benchmark" in completed.stdout
    assert "doctor" in completed.stdout
    assert "benchmark" in completed.stdout


def test_benchmark_defaults_to_qwen_7b() -> None:
    args = build_parser(PROJECT_ROOT).parse_args(
        ["benchmark", "--task", "hotpotqa", "--method", "react"]
    )

    assert args.model == DEFAULT_MODEL == "Qwen/Qwen2.5-7B-Instruct"


def test_react_best_effort_finalization_is_explicitly_opt_in() -> None:
    default_args = build_parser(PROJECT_ROOT).parse_args(
        ["benchmark", "--task", "hotpotqa", "--method", "react"]
    )
    opted_in_args = build_parser(PROJECT_ROOT).parse_args(
        [
            "benchmark",
            "--task",
            "hotpotqa",
            "--method",
            "react",
            "--react-best-effort-finalization",
        ]
    )

    assert default_args.react_best_effort_finalization is False
    assert opted_in_args.react_best_effort_finalization is True


@pytest.mark.parametrize("method", ["cot-sc", "react-cot-sc", "cot-sc-react"])
def test_cot_sc_methods_expose_paper_defaults(method: str) -> None:
    args = build_parser(PROJECT_ROOT).parse_args(
        ["benchmark", "--task", "hotpotqa", "--method", method]
    )

    assert args.cot_sc_samples == 21
    assert args.cot_sc_temperature == pytest.approx(0.7)


def test_hybrid_cot_sc_settings_are_overridable() -> None:
    args = build_parser(PROJECT_ROOT).parse_args(
        [
            "benchmark",
            "--task",
            "hotpotqa",
            "--method",
            "react-cot-sc",
            "--cot-sc-samples",
            "5",
            "--cot-sc-temperature",
            "0.9",
        ]
    )

    assert args.cot_sc_samples == 5
    assert args.cot_sc_temperature == pytest.approx(0.9)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("batch_size", [2, 3])
def test_all_methods_accept_configurable_batch_size(
    method: str,
    batch_size: int,
) -> None:
    args = build_parser(PROJECT_ROOT).parse_args(
        [
            "benchmark",
            "--task",
            "hotpotqa",
            "--method",
            method,
            "--batch-size",
            str(batch_size),
        ]
    )

    assert args.batch_size == batch_size


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_accept_fever_with_batching(method: str) -> None:
    args = build_parser(PROJECT_ROOT).parse_args(
        [
            "benchmark",
            "--task",
            "fever",
            "--method",
            method,
            "--batch-size",
            "3",
        ]
    )
    assert args.task == "fever"
    assert args.batch_size == 3


def test_fever_task_runtime_uses_paper_defaults() -> None:
    config = load_project_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        project_root=PROJECT_ROOT,
    )
    runtime = _resolve_task_runtime(config, "fever")
    assert runtime.default_max_steps == 5
    assert runtime.split == "dev"
    assert runtime.subset == "paper_dev"
    assert runtime.dataset_revision == config.fever.revision
    assert runtime.prompt_suite.version == "fever-appendix-c2-v1"
    assert runtime.guard_repeated_search is False


def test_doctor_loads_default_config() -> None:
    clean_environment = os.environ.copy()
    clean_environment.pop("REACT_LOG_LEVEL", None)
    clean_environment.pop("REACT_OUTPUT_DIR", None)

    completed = subprocess.run(
        [sys.executable, "main.py", "doctor"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        env=clean_environment,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Project configuration is healthy." in completed.stdout
    assert str(PROJECT_ROOT / "outputs") in completed.stdout
    assert "hotpotqa/hotpot_qa/distractor split=validation" in completed.stdout
