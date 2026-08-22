"""Command-line interface for reproducible experiments."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from react_reproduction import __version__
from react_reproduction.config import ProjectConfig, load_project_config
from react_reproduction.datasets.hotpotqa import load_hotpotqa
from react_reproduction.experiments.artifacts import create_run_artifacts, write_config
from react_reproduction.experiments.runner import BenchmarkRunConfig, run_hotpotqa_benchmark
from react_reproduction.llm.huggingface import HuggingFaceProvider
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
        help="Run a prompting benchmark and persist reproducible artifacts.",
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
    benchmark.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device; auto prefers CUDA when available.",
    )
    benchmark.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repositories to provide custom Python code.",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
) -> int:
    """Parse arguments and dispatch a project command."""
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
        try:
            return _run_benchmark(args, config, resolved_root)
        except (RuntimeError, ValueError, OSError) as error:
            logging.getLogger("react_reproduction").exception(
                "Benchmark failed: %s", error
            )
            return 1

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


def _run_benchmark(
    args: argparse.Namespace,
    config: ProjectConfig,
    project_root: Path,
) -> int:
    if args.task != "hotpotqa":
        raise ValueError("Only HotpotQA is implemented through Sprint 4; FEVER is Sprint 6.")
    if args.method not in {"standard", "cot", "act", "react"}:
        raise ValueError(
            f"Method {args.method!r} is not implemented through Sprint 4."
        )

    num_samples = args.num_samples or config.benchmark.num_samples
    seed = args.seed if args.seed is not None else config.benchmark.seed
    max_agent_steps = args.max_agent_steps or config.benchmark.max_agent_steps
    generation = type(config.generation)(
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.generation.temperature
        ),
        top_p=args.top_p if args.top_p is not None else config.generation.top_p,
        max_new_tokens=args.max_new_tokens or config.generation.max_new_tokens,
    )
    run_config = BenchmarkRunConfig(
        model=args.model,
        dataset=(
            f"{config.hotpotqa.dataset_name}:"
            f"{config.hotpotqa.subset}:{config.hotpotqa.split}"
        ),
        task=args.task,
        method=args.method,
        num_samples=num_samples,
        seed=seed,
        generation=asdict(generation),
        max_agent_steps=max_agent_steps,
        device=args.device,
    )
    artifacts = create_run_artifacts(
        config.output_dir,
        task=args.task,
        method=args.method,
        timestamp=run_config.timestamp,
    )
    write_config(artifacts, run_config.to_dict())
    logger = configure_logging(
        args.log_level or config.log_level,
        log_file=artifacts.run_log_path,
        quiet=args.quiet,
    )
    logger.info("Starting benchmark: task=%s method=%s", args.task, args.method)
    logger.info("Run directory: %s", artifacts.run_directory)
    cache_dir = project_root / "cache" / "huggingface"
    _configure_huggingface_cache(cache_dir)
    logger.info("Loading %d HotpotQA samples with seed=%d", num_samples, seed)
    examples = load_hotpotqa(
        num_samples,
        seed,
        dataset_name=config.hotpotqa.dataset_name,
        subset=config.hotpotqa.subset,
        split=config.hotpotqa.split,
        cache_dir=cache_dir / "datasets",
    )
    llm = HuggingFaceProvider(
        args.model,
        device=args.device,
        cache_dir=cache_dir / "hub",
        seed=seed,
        trust_remote_code=args.trust_remote_code,
    )
    if args.method == "standard":
        from react_reproduction.agents.standard import StandardAgent

        agent = StandardAgent(llm, generation)
    elif args.method == "cot":
        from react_reproduction.agents.cot import CoTAgent

        agent = CoTAgent(llm, generation)
    else:
        from react_reproduction.tools.wikipedia import (
            WikipediaClient,
            WikipediaEnvironment,
        )

        environment = WikipediaEnvironment(
            WikipediaClient(),
            max_steps=max_agent_steps,
        )
        if args.method == "act":
            from react_reproduction.agents.act import ActOnlyAgent

            agent = ActOnlyAgent(
                llm,
                generation,
                environment,
                max_steps=max_agent_steps,
            )
        else:
            from react_reproduction.agents.react import ReActAgent

            agent = ReActAgent(
                llm,
                generation,
                environment,
                max_steps=max_agent_steps,
            )

    result = run_hotpotqa_benchmark(
        examples,
        agent,
        run_config,
        output_root=config.output_dir,
        logger=logger,
        artifacts=artifacts,
        show_trajectories=args.show_trajectories,
    )
    logger.info("Benchmark complete: exact_match=%.4f", result.metrics.exact_match)
    logger.info("Metrics: %s", result.artifacts.metrics_path)
    return 0


def _configure_huggingface_cache(cache_dir: Path) -> None:
    """Keep default model/dataset downloads inside the ignored project cache."""
    resolved_cache = cache_dir.resolve()
    os.environ.setdefault("HF_HOME", str(resolved_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(resolved_cache / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(resolved_cache / "datasets"))
