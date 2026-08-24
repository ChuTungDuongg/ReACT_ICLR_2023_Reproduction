"""Command-line interface for reproducible experiments."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Sequence

from react_reproduction import __version__
from react_reproduction.config import ProjectConfig, load_project_config
from react_reproduction.agents.parsing import (
    parse_explicit_final_answer,
    parse_final_answer,
)
from react_reproduction.datasets.fever import load_fever
from react_reproduction.datasets.hotpotqa import load_hotpotqa
from react_reproduction.evaluation.fever import (
    normalize_fever_label,
    parse_fever_label,
)
from react_reproduction.evaluation.metrics import normalize_answer
from react_reproduction.experiments.artifacts import create_run_artifacts, write_config
from react_reproduction.experiments.runner import (
    BenchmarkRunConfig,
    run_fever_benchmark,
    run_hotpotqa_benchmark,
)
from react_reproduction.llm.huggingface import HuggingFaceProvider
from react_reproduction.logging_utils import configure_logging
from react_reproduction.prompts.registry import PromptSuite, get_prompt_suite


TASKS = ("hotpotqa", "fever")
METHODS = (
    "standard",
    "cot",
    "cot-sc",
    "act",
    "react",
    "react-cot-sc",
    "cot-sc-react",
)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True, slots=True)
class TaskRuntime:
    dataset_name: str
    subset: str
    split: str
    default_max_steps: int
    prompt_suite: PromptSuite
    loader: object
    runner: object
    answer_parser: object
    vote_parser: object
    answer_normalizer: object
    guard_repeated_search: bool
    dataset_revision: str | None = None


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
    benchmark.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL}).",
    )
    benchmark.add_argument("--num-samples", type=_positive_int)
    benchmark.add_argument("--seed", type=int)
    benchmark.add_argument("--temperature", type=float)
    benchmark.add_argument("--top-p", type=float)
    benchmark.add_argument("--max-new-tokens", type=_positive_int)
    benchmark.add_argument("--max-agent-steps", type=_positive_int)
    benchmark.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help=(
            "Number of examples generated together in one model batch "
            "(default: 1; try 2 or 3 on A100)."
        ),
    )
    benchmark.add_argument(
        "--react-best-effort-finalization",
        action="store_true",
        help=(
            "Force ReAct to spend its final/recovery turn on Finish. Disabled "
            "by default for paper-style evaluation."
        ),
    )
    benchmark.add_argument(
        "--cot-sc-samples",
        type=_positive_int,
        default=21,
        help="CoT samples used by CoT-SC and hybrid methods (default: 21).",
    )
    benchmark.add_argument(
        "--cot-sc-temperature",
        type=_positive_float,
        default=0.7,
        help="Sampling temperature for CoT-SC generations (default: 0.7).",
    )
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
        logger.info(
            "FEVER: %s/%s split=%s revision=%s",
            config.fever.dataset_name,
            config.fever.subset,
            config.fever.split,
            config.fever.revision,
        )
        logger.info("Paper agent step defaults: HotpotQA=7 FEVER=5")
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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _run_benchmark(
    args: argparse.Namespace,
    config: ProjectConfig,
    project_root: Path,
) -> int:
    if args.method not in set(METHODS):
        raise ValueError(f"Method {args.method!r} is not implemented.")
    react_methods = {"react", "react-cot-sc", "cot-sc-react"}
    hybrid_methods = {"react-cot-sc", "cot-sc-react"}
    if args.react_best_effort_finalization and args.method not in react_methods:
        raise ValueError(
            "--react-best-effort-finalization only applies to methods that "
            "contain ReAct."
        )
    num_samples = args.num_samples or config.benchmark.num_samples
    seed = args.seed if args.seed is not None else config.benchmark.seed
    task_runtime = _resolve_task_runtime(config, args.task)
    max_agent_steps = args.max_agent_steps or task_runtime.default_max_steps
    generation = type(config.generation)(
        temperature=(
            args.temperature
            if args.temperature is not None
            else config.generation.temperature
        ),
        top_p=args.top_p if args.top_p is not None else config.generation.top_p,
        max_new_tokens=args.max_new_tokens or config.generation.max_new_tokens,
    )
    method_settings: dict[str, object] = {}
    if args.method == "cot-sc" or args.method in hybrid_methods:
        method_settings.update(
            cot_sc_samples=args.cot_sc_samples,
            cot_sc_temperature=args.cot_sc_temperature,
            cot_sc_threshold=args.cot_sc_samples / 2,
        )
    if args.method in hybrid_methods:
        method_settings["cot_sc_fallback_threshold"] = args.cot_sc_samples / 2
    if args.method in react_methods:
        method_settings["react_best_effort_finalization"] = (
            args.react_best_effort_finalization
        )

    run_config = BenchmarkRunConfig(
        model=args.model,
        dataset=(
            f"{task_runtime.dataset_name}:"
            f"{task_runtime.subset}:{task_runtime.split}"
        ),
        task=args.task,
        method=args.method,
        num_samples=num_samples,
        seed=seed,
        generation=asdict(generation),
        max_agent_steps=max_agent_steps,
        batch_size=args.batch_size,
        method_settings=method_settings,
        device=args.device,
        dataset_split=task_runtime.split,
        dataset_subset=task_runtime.subset,
        dataset_revision=task_runtime.dataset_revision,
        prompt_version=task_runtime.prompt_suite.version,
        sampling_metadata={
            "algorithm": "python_random_sample_without_replacement",
            "seed": seed,
            "num_samples": num_samples,
        },
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
    logger.info("Code version: %s", __version__)
    logger.info("Run directory: %s", artifacts.run_directory)
    logger.info("Inference batch size: %d", args.batch_size)
    cache_dir = project_root / "cache" / "huggingface"
    _configure_huggingface_cache(cache_dir)
    logger.info(
        "Loading %d %s samples with seed=%d",
        num_samples,
        args.task,
        seed,
    )
    examples = task_runtime.loader(
        num_samples,
        seed,
        dataset_name=task_runtime.dataset_name,
        subset=task_runtime.subset,
        split=task_runtime.split,
        cache_dir=cache_dir / "datasets",
    )
    llm = HuggingFaceProvider(
        args.model,
        device=args.device,
        cache_dir=cache_dir / "hub",
        seed=seed,
        trust_remote_code=args.trust_remote_code,
    )
    run_config = replace(run_config, model_revision=llm.model_revision)
    write_config(
        artifacts,
        {
            **run_config.to_dict(),
            "sample_ids": [example.example_id for example in examples],
        },
    )
    if args.method == "standard":
        from react_reproduction.agents.standard import StandardAgent

        agent = StandardAgent(
            llm,
            generation,
            prompt_builder=task_runtime.prompt_suite.standard,
            answer_parser=task_runtime.answer_parser,
            invalid_termination_reason=(
                "invalid_label" if args.task == "fever" else "parsing_error"
            ),
        )
    elif args.method == "cot":
        from react_reproduction.agents.cot import CoTAgent

        agent = CoTAgent(
            llm,
            generation,
            prompt_builder=task_runtime.prompt_suite.cot,
            answer_parser=task_runtime.answer_parser,
            invalid_termination_reason=(
                "invalid_label" if args.task == "fever" else "parsing_error"
            ),
        )
    elif args.method == "cot-sc":
        from react_reproduction.agents.cot_sc import CoTSCAgent

        cot_sc_generation = type(config.generation)(
            temperature=args.cot_sc_temperature,
            top_p=generation.top_p,
            max_new_tokens=generation.max_new_tokens,
        )
        agent = CoTSCAgent(
            llm,
            cot_sc_generation,
            num_samples=args.cot_sc_samples,
            prompt_builder=task_runtime.prompt_suite.cot,
            answer_parser=task_runtime.vote_parser,
            answer_normalizer=task_runtime.answer_normalizer,
        )
    else:
        from react_reproduction.tools.wikipedia import (
            WikipediaClient,
            WikipediaEnvironment,
        )

        def create_environment() -> WikipediaEnvironment:
            return WikipediaEnvironment(
                WikipediaClient(),
                max_steps=max_agent_steps,
                guard_repeated_search=task_runtime.guard_repeated_search,
            )

        environment = create_environment()
        if args.method == "act":
            from react_reproduction.agents.act import ActOnlyAgent

            agent = ActOnlyAgent(
                llm,
                generation,
                environment,
                max_steps=max_agent_steps,
                environment_factory=create_environment,
                prompt_builder=task_runtime.prompt_suite.act,
                answer_normalizer=task_runtime.answer_normalizer,
            )
        elif args.method == "react":
            from react_reproduction.agents.react import ReActAgent

            agent = ReActAgent(
                llm,
                generation,
                environment,
                max_steps=max_agent_steps,
                best_effort_finalization=args.react_best_effort_finalization,
                environment_factory=create_environment,
                prompt_builder=task_runtime.prompt_suite.react,
                answer_normalizer=task_runtime.answer_normalizer,
            )
        else:
            from react_reproduction.agents.cot_sc import CoTSCAgent
            from react_reproduction.agents.hybrid import (
                CoTSCThenReActAgent,
                ReActThenCoTSCAgent,
            )
            from react_reproduction.agents.react import ReActAgent

            cot_sc_generation = type(config.generation)(
                temperature=args.cot_sc_temperature,
                top_p=generation.top_p,
                max_new_tokens=generation.max_new_tokens,
            )
            cot_sc_agent = CoTSCAgent(
                llm,
                cot_sc_generation,
                num_samples=args.cot_sc_samples,
                prompt_builder=task_runtime.prompt_suite.cot,
                answer_parser=task_runtime.vote_parser,
                answer_normalizer=task_runtime.answer_normalizer,
            )
            react_agent = ReActAgent(
                llm,
                generation,
                environment,
                max_steps=max_agent_steps,
                best_effort_finalization=args.react_best_effort_finalization,
                environment_factory=create_environment,
                prompt_builder=task_runtime.prompt_suite.react,
                answer_normalizer=task_runtime.answer_normalizer,
            )
            agent = (
                ReActThenCoTSCAgent(react_agent, cot_sc_agent)
                if args.method == "react-cot-sc"
                else CoTSCThenReActAgent(cot_sc_agent, react_agent)
            )

    result = task_runtime.runner(
        examples,
        agent,
        run_config,
        output_root=config.output_dir,
        logger=logger,
        artifacts=artifacts,
        show_trajectories=args.show_trajectories,
    )
    primary_metric = (
        result.metrics.accuracy
        if args.task == "fever"
        else result.metrics.exact_match
    )
    logger.info("Benchmark complete: primary_metric=%.4f", primary_metric)
    logger.info("Metrics: %s", result.artifacts.metrics_path)
    return 0


def _resolve_task_runtime(config: ProjectConfig, task: str) -> TaskRuntime:
    if task == "hotpotqa":
        return TaskRuntime(
            dataset_name=config.hotpotqa.dataset_name,
            subset=config.hotpotqa.subset,
            split=config.hotpotqa.split,
            default_max_steps=config.max_agent_steps_for(task),
            prompt_suite=get_prompt_suite(task),
            loader=load_hotpotqa,
            runner=run_hotpotqa_benchmark,
            answer_parser=parse_final_answer,
            vote_parser=parse_explicit_final_answer,
            answer_normalizer=normalize_answer,
            guard_repeated_search=True,
        )
    if task == "fever":
        return TaskRuntime(
            dataset_name=config.fever.dataset_name,
            subset=config.fever.subset,
            split=config.fever.split,
            default_max_steps=config.max_agent_steps_for(task),
            prompt_suite=get_prompt_suite(task),
            loader=partial(
                load_fever,
                source_revision=config.fever.revision,
            ),
            runner=run_fever_benchmark,
            answer_parser=parse_fever_label,
            vote_parser=parse_fever_label,
            answer_normalizer=normalize_fever_label,
            guard_repeated_search=False,
            dataset_revision=config.fever.revision,
        )
    raise ValueError(f"Unsupported task: {task!r}.")


def _configure_huggingface_cache(cache_dir: Path) -> None:
    """Keep default model/dataset downloads inside the ignored project cache."""
    resolved_cache = cache_dir.resolve()
    os.environ.setdefault("HF_HOME", str(resolved_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(resolved_cache / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(resolved_cache / "datasets"))
