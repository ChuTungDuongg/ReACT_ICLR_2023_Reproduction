"""Executable entry point for the ReAct paper reproduction project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"

# Keep `python main.py ...` usable after a plain Git clone without requiring an
# editable package installation.
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from react_reproduction.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(project_root=PROJECT_ROOT))
