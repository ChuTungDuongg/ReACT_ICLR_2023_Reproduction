"""Network- and model-free smoke tests for the repository bootstrap."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
    assert "Project bootstrap is healthy." in completed.stdout
    assert str(PROJECT_ROOT / "outputs") in completed.stdout
