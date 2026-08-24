"""Shared runner serialization and Accuracy tests for FEVER."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from react_reproduction.agents.base import AgentResult, TrajectoryStep
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.experiments.runner import (
    BenchmarkRunConfig,
    MockPredictor,
    run_fever_benchmark,
)


def _config(*, method: str, count: int) -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        model="mock",
        dataset="ysymyth/ReAct:paper_dev:dev",
        task="fever",
        method=method,
        num_samples=count,
        seed=42,
        generation={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32},
        max_agent_steps=5,
        batch_size=2,
        dataset_subset="paper_dev",
        dataset_split="dev",
        dataset_revision="6bdb3a1",
        prompt_version="fever-appendix-c2-v1",
        sampling_metadata={"seed": 42, "num_samples": count},
        timestamp="2026-08-24T16:00:00+00:00",
    )


def test_fever_runner_persists_claims_accuracy_and_invalid_counts(tmp_path) -> None:
    examples = [
        BenchmarkExample("1", "Claim one.", "SUPPORTS"),
        BenchmarkExample("2", "Claim two.", "REFUTES"),
        BenchmarkExample("3", "Claim three.", "NOT ENOUGH INFO"),
        BenchmarkExample("4", "Claim four.", "SUPPORTS"),
    ]
    predictor = MockPredictor(
        {
            "1": "supports",
            "2": "SUPPORTS",
            "3": "not enough information",
            "4": "probably supports",
        }
    )
    stream = StringIO()
    logger = logging.getLogger("test.fever.runner")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))

    result = run_fever_benchmark(
        examples,
        predictor,
        _config(method="standard", count=4),
        output_root=tmp_path,
        logger=logger,
    )
    metrics = json.loads(result.artifacts.metrics_path.read_text("utf-8"))
    config = json.loads(result.artifacts.config_path.read_text("utf-8"))
    predictions = [
        json.loads(line)
        for line in result.artifacts.predictions_path.read_text("utf-8").splitlines()
    ]

    assert result.metrics.accuracy == pytest.approx(0.5)
    assert metrics["primary_metric"] == {"name": "accuracy", "value": 0.5}
    assert metrics["correct"] == 2
    assert metrics["incorrect"] == 2
    assert metrics["invalid_unparsed"] == 1
    assert metrics["confusion_matrix"]["SUPPORTS"]["INVALID"] == 1
    assert config["sample_ids"] == ["1", "2", "3", "4"]
    assert config["max_agent_steps"] == 5
    assert config["prompt_version"] == "fever-appendix-c2-v1"
    assert [record["claim"] for record in predictions] == [
        "Claim one.",
        "Claim two.",
        "Claim three.",
        "Claim four.",
    ]
    assert predictions[-1]["prediction"] == ""
    assert predictions[-1]["invalid"] is True
    assert "Running Accuracy: 50.00%" in stream.getvalue()
    assert "=== FINAL FEVER METRICS ===" in stream.getvalue()


def test_fever_runner_serializes_trajectory_and_vote_metadata(tmp_path) -> None:
    example = BenchmarkExample("1", "A claim.", "SUPPORTS")

    class Predictor:
        def predict(self, item: BenchmarkExample) -> AgentResult:
            return AgentResult(
                prediction="SUPPORTS",
                steps=1,
                tool_calls=0,
                termination_reason="cot_sc_majority",
                trajectory=(
                    TrajectoryStep(
                        step_index=1,
                        model_output="Answer: SUPPORTS",
                        thought="Evidence supports it.",
                        phase="cot_sc",
                    ),
                ),
                metadata={
                    "cot_sc_samples": 21,
                    "winning_label": "SUPPORTS",
                    "winning_count": 13,
                    "winning_fraction": 13 / 21,
                    "valid_sample_count": 21,
                    "vote_distribution": {"SUPPORTS": 13, "REFUTES": 8},
                    "paper_threshold_met": True,
                },
            )

    result = run_fever_benchmark(
        [example],
        Predictor(),
        _config(method="cot-sc", count=1),
        output_root=tmp_path,
    )
    prediction = json.loads(result.artifacts.predictions_path.read_text("utf-8"))
    trajectory = json.loads(result.artifacts.trajectories_path.read_text("utf-8"))
    assert prediction["winning_count"] == 13
    assert prediction["paper_threshold_met"] is True
    assert trajectory["steps"][0]["phase"] == "cot_sc"
