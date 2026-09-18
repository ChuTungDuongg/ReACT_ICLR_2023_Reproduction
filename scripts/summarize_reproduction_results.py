"""Select, validate, summarize, and plot completed reproduction runs.

The selector prefers the largest compatible completed run for each task/method
and uses the newest timestamp only as a tie-breaker. This prevents a later
smoke test from replacing a full benchmark in the published summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
METHOD_LABELS = {
    "standard": "Standard",
    "cot": "CoT",
    "cot-sc": "CoT-SC",
    "act": "Act",
    "react": "ReAct",
    "react-cot-sc": "ReAct -> CoT-SC",
    "cot-sc-react": "CoT-SC -> ReAct",
}
PRIMARY_METRICS = {"hotpotqa": "exact_match", "fever": "accuracy"}
REQUIRED_FILES = ("config.json", "metrics.json", "predictions.jsonl", "run.log")


@dataclass(frozen=True)
class Run:
    task: str
    method: str
    directory: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]
    trajectories: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path}:{line_number}")
        rows.append(value)
    return tuple(rows)


def _is_completed(directory: Path) -> bool:
    if any(not (directory / name).is_file() for name in REQUIRED_FILES):
        return False
    log_text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
    if " ERROR " in log_text or "Traceback (most recent call last)" in log_text:
        return False
    return "Metrics:" in log_text


def _load_candidate(task: str, method: str, directory: Path) -> Run | None:
    if not _is_completed(directory):
        return None
    config = _read_json(directory / "config.json")
    metrics = _read_json(directory / "metrics.json")
    predictions = _read_jsonl(directory / "predictions.jsonl")
    trajectory_path = directory / "trajectories.jsonl"
    trajectories = _read_jsonl(trajectory_path) if trajectory_path.is_file() else ()
    if config.get("task") != task or config.get("method") != method:
        return None
    expected = config.get("num_samples")
    total = metrics.get("total_examples")
    if not isinstance(expected, int) or expected <= 0:
        return None
    if total != expected or len(predictions) != expected:
        return None
    ids = [str(row.get("example_id")) for row in predictions]
    if len(set(ids)) != expected:
        return None
    sample_ids = config.get("sample_ids")
    if sample_ids is not None and [str(value) for value in sample_ids] != ids:
        return None
    return Run(task, method, directory, config, metrics, predictions, trajectories)


def select_runs(outputs_root: Path) -> list[Run]:
    selected: list[Run] = []
    for task in TASKS:
        for method in METHODS:
            method_dir = outputs_root / task / method
            candidates = []
            if method_dir.is_dir():
                for directory in method_dir.iterdir():
                    if directory.is_dir():
                        candidate = _load_candidate(task, method, directory)
                        if candidate is not None:
                            candidates.append(candidate)
            if not candidates:
                raise RuntimeError(f"No completed compatible run for {task}/{method}")
            candidates.sort(
                key=lambda run: (
                    int(run.config["num_samples"]),
                    str(run.config.get("timestamp", run.directory.name)),
                )
            )
            selected.append(candidates[-1])
    return selected


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else None


def _path_summaries(run: Run) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in run.predictions:
        path = prediction.get("execution_path")
        if not isinstance(path, str):
            metadata = prediction.get("agent_metadata")
            if isinstance(metadata, dict):
                path = metadata.get("execution_path")
        if not isinstance(path, str) and run.method in {
            "react-cot-sc",
            "cot-sc-react",
        }:
            termination = prediction.get("termination_reason")
            if isinstance(termination, str):
                path = termination
        if isinstance(path, str):
            grouped[path].append(prediction)
    if not grouped:
        return None
    result: dict[str, Any] = {}
    for path, rows in sorted(grouped.items()):
        correct = sum(bool(row.get("correct")) for row in rows)
        entry: dict[str, Any] = {
            "num_examples": len(rows),
            "correct": correct,
            "accuracy_or_exact_match": correct / len(rows),
            "average_steps": _mean(rows, "steps"),
            "average_tool_calls": _mean(rows, "tool_calls"),
        }
        f1 = _mean(rows, "answer_f1")
        if f1 is not None:
            entry["f1"] = f1
        invalid = sum(bool(row.get("invalid")) for row in rows if "invalid" in row)
        if any("invalid" in row for row in rows):
            entry["invalid_unparsed"] = invalid
        result[path] = entry
    return result


def _termination_summaries(run: Run) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in run.predictions:
        grouped[str(prediction.get("termination_reason", "missing"))].append(prediction)
    result: dict[str, Any] = {}
    for reason, rows in sorted(grouped.items()):
        correct = sum(bool(row.get("correct")) for row in rows)
        entry: dict[str, Any] = {
            "num_examples": len(rows),
            "correct": correct,
            "accuracy_or_exact_match": correct / len(rows),
        }
        f1 = _mean(rows, "answer_f1")
        if f1 is not None:
            entry["f1"] = f1
        result[reason] = entry
    return result


def _trajectory_summary(run: Run) -> dict[str, Any] | None:
    if not run.trajectories:
        return None
    action_counts: Counter[str] = Counter()
    invalid_action_steps = 0
    action_only_outputs = 0
    failed_search_observations = 0
    lookup_no_match_steps = 0
    repeated_search_examples = 0
    examples_with_invalid_action = 0
    for trajectory in run.trajectories:
        searches: list[str] = []
        has_invalid = False
        for step in trajectory.get("steps", []):
            action = step.get("action")
            if isinstance(action, str) and "[" in action:
                action_type = action.split("[", 1)[0]
                action_counts[action_type] += 1
                if action_type == "Search":
                    searches.append(action.casefold())
            observation = str(step.get("observation") or "")
            if observation.startswith("Invalid action format:"):
                invalid_action_steps += 1
                has_invalid = True
            if observation.startswith("Could not find"):
                failed_search_observations += 1
            if observation.startswith("No match for"):
                lookup_no_match_steps += 1
            model_output = str(step.get("model_output") or "").strip()
            if re.fullmatch(r"(?:Search|Lookup|Finish)\[[\s\S]*\]", model_output):
                action_only_outputs += 1
        if len(searches) != len(set(searches)):
            repeated_search_examples += 1
        if has_invalid:
            examples_with_invalid_action += 1
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "invalid_action_steps": invalid_action_steps,
        "examples_with_invalid_action": examples_with_invalid_action,
        "action_only_outputs": action_only_outputs,
        "failed_search_observations": failed_search_observations,
        "lookup_no_match_steps": lookup_no_match_steps,
        "examples_with_repeated_search": repeated_search_examples,
    }


def _config_value(config: dict[str, Any], name: str) -> Any:
    if name in config and config[name] is not None:
        return config[name]
    return config.get("method_settings", {}).get(name)


def summarize_run(run: Run, repository_root: Path) -> dict[str, Any]:
    config = run.config
    metrics = run.metrics
    primary_name = PRIMARY_METRICS[run.task]
    entry: dict[str, Any] = {
        "task": run.task,
        "method": run.method,
        "method_label": METHOD_LABELS[run.method],
        "run_directory": run.directory.relative_to(repository_root).as_posix(),
        "num_examples": metrics["total_examples"],
        "primary_metric": {
            "name": primary_name,
            "value": metrics[primary_name],
        },
        "correct": metrics.get("correct"),
        "incorrect": metrics.get("incorrect"),
        "average_steps": metrics.get("average_steps"),
        "average_tool_calls": metrics.get("average_tool_calls"),
        "runtime_seconds": metrics.get("runtime"),
        "termination_reasons": metrics.get("termination_reasons"),
        "configuration": {
            "model": config.get("model"),
            "model_revision": config.get("model_revision"),
            "dataset": config.get("dataset"),
            "dataset_revision": config.get("dataset_revision"),
            "dataset_subset": config.get("dataset_subset", config.get("subset")),
            "dataset_split": config.get("dataset_split", config.get("split")),
            "seed": config.get("seed"),
            "batch_size": config.get("batch_size"),
            "max_agent_steps": config.get("max_agent_steps"),
            "generation_temperature": config.get("generation", {}).get("temperature"),
            "generation_top_p": config.get("generation", {}).get("top_p"),
            "generation_max_new_tokens": config.get("generation", {}).get(
                "max_new_tokens"
            ),
            "cot_sc_samples": _config_value(config, "cot_sc_samples"),
            "cot_sc_temperature": _config_value(config, "cot_sc_temperature"),
            "cot_sc_threshold": _config_value(config, "cot_sc_threshold")
            or _config_value(config, "cot_sc_fallback_threshold"),
            "react_best_effort_finalization": _config_value(
                config, "react_best_effort_finalization"
            ),
            "prompt_version": config.get("prompt_version"),
            "code_version": config.get("code_version"),
        },
        "termination_outcomes": _termination_summaries(run),
    }
    for name in (
        "accuracy",
        "exact_match",
        "f1",
        "precision",
        "recall",
        "invalid_unparsed",
        "per_class_accuracy",
        "confusion_matrix",
        "gold_label_distribution",
        "predicted_label_distribution",
    ):
        if name in metrics:
            entry[name] = metrics[name]
    paths = _path_summaries(run)
    if paths is not None:
        entry["execution_paths"] = paths
    trajectory = _trajectory_summary(run)
    if trajectory is not None:
        entry["trajectory_analysis"] = trajectory
    return entry


def build_summary(outputs_root: Path, repository_root: Path) -> dict[str, Any]:
    selected = select_runs(outputs_root)
    id_order: dict[str, list[str]] = {}
    validation: dict[str, Any] = {}
    for task in TASKS:
        task_runs = [run for run in selected if run.task == task]
        first_ids = [str(row["example_id"]) for row in task_runs[0].predictions]
        id_order[task] = first_ids
        validation[task] = {
            "all_seven_methods_present": len(task_runs) == len(METHODS),
            "same_example_order_across_methods": all(
                [str(row["example_id"]) for row in run.predictions] == first_ids
                for run in task_runs[1:]
            ),
            "selected_example_count": len(first_ids),
        }
    return {
        "selection_rule": (
            "For each task/method, select completed compatible runs with matching "
            "config, metrics, prediction count, unique IDs, and a terminal Metrics "
            "log entry; prefer the largest run, then the newest timestamp."
        ),
        "method_semantics": {
            "react-cot-sc": "ReAct -> CoT-SC",
            "cot-sc-react": "CoT-SC -> ReAct",
        },
        "validation": validation,
        "runs": [summarize_run(run, repository_root) for run in selected],
    }


CSV_FIELDS = (
    "task",
    "method",
    "method_label",
    "num_examples",
    "primary_metric_name",
    "primary_metric_value",
    "accuracy",
    "exact_match",
    "f1",
    "invalid_unparsed",
    "average_steps",
    "average_tool_calls",
    "runtime_seconds",
    "termination_reasons",
    "run_directory",
)


def _csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in summary["runs"]:
        rows.append(
            {
                "task": run["task"],
                "method": run["method"],
                "method_label": run["method_label"],
                "num_examples": run["num_examples"],
                "primary_metric_name": run["primary_metric"]["name"],
                "primary_metric_value": run["primary_metric"]["value"],
                "accuracy": run.get("accuracy", ""),
                "exact_match": run.get("exact_match", ""),
                "f1": run.get("f1", ""),
                "invalid_unparsed": run.get("invalid_unparsed", ""),
                "average_steps": run.get("average_steps", ""),
                "average_tool_calls": run.get("average_tool_calls", ""),
                "runtime_seconds": run.get("runtime_seconds", ""),
                "termination_reasons": json.dumps(
                    run.get("termination_reasons", {}), sort_keys=True
                ),
                "run_directory": run["run_directory"],
            }
        )
    return rows


def write_summaries(summary: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "results_summary.json"
    csv_path = reports_dir / "results_summary.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_rows(summary))


def _draw_bar_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    color: str,
) -> None:
    try:
        from reportlab.lib.colors import HexColor
        from reportlab.pdfgen import canvas
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("ReportLab is required to generate figures") from error

    width, height = 540, 300
    chart = canvas.Canvas(str(path), pagesize=(width, height), invariant=1)
    chart.setTitle(title)
    chart.setFont("Helvetica-Bold", 13)
    chart.drawString(48, 274, title)
    chart.setFont("Helvetica", 8)
    chart.setFillColor(HexColor("#4B5563"))
    chart.drawString(48, 260, subtitle)
    left, bottom, plot_width, plot_height = 52, 62, 466, 176
    maximum = max(values) if values else 1.0
    ceiling = max(1.0, maximum * 1.18)
    chart.setStrokeColor(HexColor("#D1D5DB"))
    chart.setFillColor(HexColor("#374151"))
    chart.setFont("Helvetica", 7)
    for index in range(6):
        value = ceiling * index / 5
        y = bottom + plot_height * index / 5
        chart.line(left, y, left + plot_width, y)
        chart.drawRightString(left - 5, y - 2, f"{value:.0f}")
    chart.saveState()
    chart.translate(14, bottom + plot_height / 2)
    chart.rotate(90)
    chart.drawCentredString(0, 0, y_label)
    chart.restoreState()
    slot = plot_width / len(values)
    bar_width = slot * 0.58
    chart.setFillColor(HexColor(color))
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = left + index * slot + (slot - bar_width) / 2
        height_value = plot_height * value / ceiling
        chart.rect(x, bottom, bar_width, height_value, stroke=0, fill=1)
        chart.setFillColor(HexColor("#111827"))
        chart.setFont("Helvetica-Bold", 7)
        chart.drawCentredString(x + bar_width / 2, bottom + height_value + 5, f"{value:.1f}")
        chart.saveState()
        chart.translate(x + bar_width / 2 + 2, bottom - 7)
        chart.rotate(35)
        chart.setFont("Helvetica", 6.5)
        chart.drawRightString(0, 0, label)
        chart.restoreState()
        chart.setFillColor(HexColor(color))
    chart.showPage()
    chart.save()


def _draw_grouped_chart(path: Path, summary: dict[str, Any]) -> None:
    try:
        from reportlab.lib.colors import HexColor
        from reportlab.pdfgen import canvas
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("ReportLab is required to generate figures") from error

    by_key = {(run["task"], run["method"]): run for run in summary["runs"]}
    width, height = 540, 300
    chart = canvas.Canvas(str(path), pagesize=(width, height), invariant=1)
    chart.setTitle("Average Wikipedia tool calls by method")
    chart.setFont("Helvetica-Bold", 13)
    chart.drawString(48, 274, "Average Wikipedia tool calls by method")
    chart.setFont("Helvetica", 8)
    chart.setFillColor(HexColor("#4B5563"))
    chart.drawString(48, 260, "Counts are means over each selected 500-example run.")
    left, bottom, plot_width, plot_height = 52, 62, 466, 176
    values = [
        float(by_key[(task, method)]["average_tool_calls"])
        for method in METHODS
        for task in TASKS
    ]
    ceiling = max(1.0, max(values) * 1.2)
    chart.setStrokeColor(HexColor("#D1D5DB"))
    chart.setFillColor(HexColor("#374151"))
    chart.setFont("Helvetica", 7)
    for index in range(6):
        value = ceiling * index / 5
        y = bottom + plot_height * index / 5
        chart.line(left, y, left + plot_width, y)
        chart.drawRightString(left - 5, y - 2, f"{value:.1f}")
    group_width = plot_width / len(METHODS)
    bar_width = group_width * 0.28
    colors = (HexColor("#2563EB"), HexColor("#D97706"))
    for method_index, method in enumerate(METHODS):
        center = left + (method_index + 0.5) * group_width
        for task_index, task in enumerate(TASKS):
            value = float(by_key[(task, method)]["average_tool_calls"])
            x = center + (task_index - 1) * bar_width
            bar_height = plot_height * value / ceiling
            chart.setFillColor(colors[task_index])
            chart.rect(x, bottom, bar_width, bar_height, stroke=0, fill=1)
        chart.saveState()
        chart.translate(center + 4, bottom - 7)
        chart.rotate(35)
        chart.setFillColor(HexColor("#111827"))
        chart.setFont("Helvetica", 6.5)
        chart.drawRightString(0, 0, METHOD_LABELS[method])
        chart.restoreState()
    chart.setFillColor(colors[0])
    chart.rect(392, 276, 8, 8, stroke=0, fill=1)
    chart.setFillColor(HexColor("#111827"))
    chart.setFont("Helvetica", 7)
    chart.drawString(404, 277, "HotpotQA")
    chart.setFillColor(colors[1])
    chart.rect(458, 276, 8, 8, stroke=0, fill=1)
    chart.setFillColor(HexColor("#111827"))
    chart.drawString(470, 277, "FEVER")
    chart.showPage()
    chart.save()


def write_figures(summary: dict[str, Any], reports_dir: Path) -> None:
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for task, filename, title, color in (
        ("hotpotqa", "hotpotqa_exact_match.pdf", "HotpotQA exact match by method", "#2563EB"),
        ("fever", "fever_accuracy.pdf", "FEVER accuracy by method", "#D97706"),
    ):
        runs = [run for run in summary["runs"] if run["task"] == task]
        _draw_bar_chart(
            figures_dir / filename,
            title=title,
            subtitle="Selected full benchmark runs; percentages over 500 examples.",
            labels=[run["method_label"] for run in runs],
            values=[100.0 * float(run["primary_metric"]["value"]) for run in runs],
            y_label="Percent",
            color=color,
        )
    _draw_grouped_chart(figures_dir / "average_tool_calls.pdf", summary)


def verify_written(summary: dict[str, Any], reports_dir: Path) -> None:
    json_path = reports_dir / "results_summary.json"
    csv_path = reports_dir / "results_summary.csv"
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    if stored != summary:
        raise RuntimeError(f"Stale or mismatched summary: {json_path}")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        actual_rows = list(csv.DictReader(handle))
    expected_rows = []
    for row in _csv_rows(summary):
        expected_rows.append({key: str(value) for key, value in row.items()})
    if actual_rows != expected_rows:
        raise RuntimeError(f"Stale or mismatched summary: {csv_path}")
    _verify_report_tables(summary, reports_dir)


def _latex_method(label: str) -> str:
    return label.replace(" -> ", " $\\rightarrow$ ")


def _markdown_method(label: str) -> str:
    return label.replace(" -> ", " → ")


def _verify_report_tables(summary: dict[str, Any], reports_dir: Path) -> None:
    markdown_by_task = {
        "hotpotqa": (reports_dir / "hotpotqa_report.md").read_text(encoding="utf-8"),
        "fever": (reports_dir / "fever_report.md").read_text(encoding="utf-8"),
    }
    latex = (reports_dir / "react_reproduction_report.tex").read_text(
        encoding="utf-8"
    )
    plain_latex = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", latex)
    readme = (reports_dir / "README.md").read_text(encoding="utf-8")
    root_readme = (reports_dir.parent / "README.md").read_text(encoding="utf-8")
    for run in summary["runs"]:
        metric = 100.0 * float(run["primary_metric"]["value"])
        runtime_minutes = float(run["runtime_seconds"]) / 60.0
        if run["task"] == "hotpotqa":
            markdown_row = (
                f'| {_markdown_method(run["method_label"])} | '
                f'{run["correct"]} / {run["num_examples"]} '
                f'| {metric:.1f} | {100.0 * float(run["f1"]):.2f} '
                f'| {float(run["average_steps"]):.3f} '
                f'| {float(run["average_tool_calls"]):.3f} | {runtime_minutes:.2f} |'
            )
            latex_row = (
                f'{_latex_method(run["method_label"])} & {run["correct"]} '
                f'& {metric:.1f} & {100.0 * float(run["f1"]):.2f} '
                f'& {float(run["average_steps"]):.3f} '
                f'& {float(run["average_tool_calls"]):.3f} '
                f'& {runtime_minutes:.2f}'
            )
        else:
            markdown_row = (
                f'| {_markdown_method(run["method_label"])} | '
                f'{run["correct"]} / {run["num_examples"]} '
                f'| {metric:.1f} | {run["invalid_unparsed"]} '
                f'| {float(run["average_steps"]):.3f} '
                f'| {float(run["average_tool_calls"]):.3f} | {runtime_minutes:.2f} |'
            )
            latex_row = (
                f'{_latex_method(run["method_label"])} & {run["correct"]} '
                f'& {metric:.1f} & {run["invalid_unparsed"]} '
                f'& {float(run["average_steps"]):.3f} '
                f'& {float(run["average_tool_calls"]):.3f} '
                f'& {runtime_minutes:.2f}'
            )
        if markdown_row not in markdown_by_task[run["task"]]:
            raise RuntimeError(f"Report table mismatch: {markdown_row}")
        if latex_row not in plain_latex:
            raise RuntimeError(f"LaTeX table mismatch: {latex_row}")
        if f'`{run["run_directory"]}`' not in readme:
            raise RuntimeError(f'Missing selected-run mapping: {run["run_directory"]}')

    by_key = {(run["task"], run["method"]): run for run in summary["runs"]}
    for method in METHODS:
        hotpot = by_key[("hotpotqa", method)]
        fever = by_key[("fever", method)]
        label = _markdown_method(hotpot["method_label"])
        if method == "react-cot-sc":
            label = f"**{label}**"
        values = (
            100.0 * float(hotpot["exact_match"]),
            100.0 * float(hotpot["f1"]),
            100.0 * float(fever["accuracy"]),
        )
        if method == "react-cot-sc":
            root_row = (
                f"| {label} | **{values[0]:.1f}** | **{values[1]:.2f}** "
                f"| **{values[2]:.1f}** | **{fever['invalid_unparsed']}** |"
            )
        else:
            root_row = (
                f"| {label} | {values[0]:.1f} | {values[1]:.2f} "
                f"| {values[2]:.1f} | {fever['invalid_unparsed']} |"
            )
        if root_row not in root_readme:
            raise RuntimeError(f"Root README benchmark table mismatch: {root_row}")

    fever_runs = [run for run in summary["runs"] if run["task"] == "fever"]
    for run in fever_runs:
        class_accuracy = run["per_class_accuracy"]
        values = (
            100.0 * float(class_accuracy["SUPPORTS"]),
            100.0 * float(class_accuracy["REFUTES"]),
            100.0 * float(class_accuracy["NOT ENOUGH INFO"]),
        )
        markdown_row = (
            f'| {_markdown_method(run["method_label"])} | {values[0]:.1f} '
            f'| {values[1]:.1f} | {values[2]:.1f} |'
        )
        latex_row = (
            f'{_latex_method(run["method_label"])} & {values[0]:.1f} '
            f'& {values[1]:.1f} & {values[2]:.1f}'
        )
        if markdown_row not in markdown_by_task["fever"]:
            raise RuntimeError(f"FEVER class table mismatch: {markdown_row}")
        if latex_row not in plain_latex:
            raise RuntimeError(f"LaTeX FEVER class table mismatch: {latex_row}")

    required_files = (
        reports_dir / "figures" / "hotpotqa_exact_match.pdf",
        reports_dir / "figures" / "fever_accuracy.pdf",
        reports_dir / "figures" / "average_tool_calls.pdf",
        reports_dir / "react_reproduction_report.pdf",
    )
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty generated artifact: {path}")

    text_files = (
        reports_dir / "README.md",
        reports_dir / "hotpotqa_report.md",
        reports_dir / "fever_report.md",
        reports_dir / "react_reproduction_report.tex",
        reports_dir.parent / "README.md",
    )
    placeholder = re.compile(r"\b(?:TODO|TBD|XXX|PLACEHOLDER)\b")
    for path in text_files:
        match = placeholder.search(path.read_text(encoding="utf-8"))
        if match:
            raise RuntimeError(f"Placeholder {match.group()} remains in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summaries-only", action="store_true")
    mode.add_argument("--figures-only", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    outputs_root = (args.outputs_root or repository_root / "outputs").resolve()
    reports_dir = (args.reports_dir or repository_root / "reports").resolve()
    summary = build_summary(outputs_root, repository_root)
    if args.check:
        verify_written(summary, reports_dir)
    elif args.figures_only:
        write_figures(summary, reports_dir)
    elif args.summaries_only:
        write_summaries(summary, reports_dir)
    else:
        write_summaries(summary, reports_dir)
        write_figures(summary, reports_dir)


if __name__ == "__main__":
    main()
