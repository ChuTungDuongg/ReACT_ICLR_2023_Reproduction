"""Command-line interface for reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from react_reproduction import __version__
from react_reproduction.config import load_project_config
from react_reproduction.logging_utils import configure_logging


TASKS = ("hotpotqa", "fever")
METHODS = ("standard", "cot", "act", "react")


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    """Build the public CLI without performing any expensive work."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Reproduce and benchmark prompting methods from the ReAct paper.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser(
        "doctor",
        help="Validate the project configuration and dataset settings.",
    )
    _add_shared_options(doctor, project_root)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run a benchmark (interface placeholder; implementation is pending).",
    )
    _add_shared_options(benchmark, project_root)
    benchmark.add_argument("--task", required=True, choices=TASKS)
    benchmark.add_argument("--method", required=True, choices=METHODS)
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--num-samples", type=_positive_int)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--temperature", type=float)
    benchmark.add_argument("--top-p", type=float)
    benchmark.add_argument("--max-new-tokens", type=_positive_int)
    benchmark.add_argument("--max-agent-steps", type=_positive_int)
    benchmark.add_argument("--show-trajectories", action="store_true")
    benchmark.add_argument("--quiet", action="store_true")

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
) -> int:
    """Parse arguments and dispatch a Sprint 0 command."""
    resolved_root = (project_root or Path.cwd()).resolve()
    parser = build_parser(resolved_root)
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    config = load_project_config(args.config, project_root=resolved_root)
    log_level = args.log_level or config.log_level
    logger = configure_logging(log_level, quiet=getattr(args, "quiet", False))

    if args.command == "doctor":
        logger.info("Project configuration is healthy.")
        logger.info("Project: %s", config.project_name)
        logger.info("Config: %s", args.config.resolve())
        logger.info("Output directory: %s", config.output_dir)
        logger.info(
            "HotpotQA: %s/%s split=%s",
            config.hotpotqa.dataset_name,
            config.hotpotqa.subset,
            config.hotpotqa.split,
        )
        return 0

    if args.command == "benchmark":
        logger.error(
            "CLI benchmark execution requires the model and prompting methods "
            "scheduled for Sprint 2. Sprint 1 provides a tested mock runner."
        )
        return 2

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _add_shared_options(parser: argparse.ArgumentParser, project_root: Path) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "default.yaml",
        help="YAML configuration path (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Override the configured log level.",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed
