"""End-to-end test of the mock runner and run serialization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.experiments.runner import (
    BenchmarkRunConfig,
    MockPredictor,
    run_hotpotqa_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mock_benchmark_serializes_config_metrics_and_predictions() -> None:
    test_output_root = PROJECT_ROOT / "outputs" / f"_pytest_{uuid4().hex}"
    examples = [
        BenchmarkExample("example-1", "First question?", "The Answer"),
        BenchmarkExample("example-2", "Second question?", "Berlin"),
    ]
    predictor = MockPredictor(
        {
            "example-1": "answer",
            "example-2": "Munich",
        }
    )
    config = BenchmarkRunConfig(
        model="mock",
        dataset="hotpotqa/hotpot_qa:distractor:validation",
        task="hotpotqa",
        method="standard",
        num_samples=2,
        seed=42,
        generation={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 16},
        max_agent_steps=7,
        timestamp="2026-08-23T01:30:00+00:00",
    )

    try:
        result = run_hotpotqa_benchmark(
            examples,
            predictor,
            config,
            output_root=test_output_root,
        )

        serialized_config = json.loads(result.artifacts.config_path.read_text("utf-8"))
        serialized_metrics = json.loads(result.artifacts.metrics_path.read_text("utf-8"))
        prediction_lines = result.artifacts.predictions_path.read_text("utf-8").splitlines()

        assert serialized_config["seed"] == 42
        assert serialized_config["model"] == "mock"
        assert serialized_metrics["exact_match"] == 0.5
        assert serialized_metrics["correct"] == 1
        assert serialized_metrics["incorrect"] == 1
        assert serialized_metrics["average_steps"] == 1.0
        assert serialized_metrics["average_tool_calls"] == 0.0
        assert serialized_metrics["termination_reasons"] == {"mock_completed": 2}
        assert len(prediction_lines) == 2
        assert json.loads(prediction_lines[0])["correct"] is True
        assert result.artifacts.run_directory.name == "2026-08-23_013000"
    finally:
        if test_output_root.exists():
            resolved_test_root = test_output_root.resolve()
            assert resolved_test_root.parent == (PROJECT_ROOT / "outputs").resolve()
            shutil.rmtree(resolved_test_root)
