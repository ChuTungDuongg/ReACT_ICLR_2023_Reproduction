"""End-to-end test of the mock runner and run serialization."""

from __future__ import annotations

import json
import logging
import shutil
from io import StringIO
from pathlib import Path
from uuid import uuid4

from react_reproduction.agents.base import AgentResult, TrajectoryStep
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
        BenchmarkExample(
            "example-1",
            "First question?",
            "The Answer",
            metadata={
                "supporting_facts": {"title": ["Article A"], "sent_id": [0]}
            },
        ),
        BenchmarkExample(
            "example-2",
            "Second question?",
            "Berlin",
            metadata={
                "supporting_facts": {"title": ["Article B"], "sent_id": [1]}
            },
        ),
    ]
    predictor = MockPredictor(
        {
            "example-1": "answer",
            "example-2": "",
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

    log_stream = StringIO()
    logger = logging.getLogger(f"test.runner.{uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(log_stream))

    try:
        result = run_hotpotqa_benchmark(
            examples,
            predictor,
            config,
            output_root=test_output_root,
            logger=logger,
        )

        serialized_config = json.loads(result.artifacts.config_path.read_text("utf-8"))
        serialized_metrics = json.loads(result.artifacts.metrics_path.read_text("utf-8"))
        prediction_lines = result.artifacts.predictions_path.read_text("utf-8").splitlines()

        assert serialized_config["seed"] == 42
        assert serialized_config["model"] == "mock"
        assert serialized_config["code_version"] == "0.7.0"
        assert serialized_metrics["exact_match"] == 0.5
        assert serialized_metrics["f1"] == 0.5
        assert serialized_metrics["precision"] == 0.5
        assert serialized_metrics["recall"] == 0.5
        assert serialized_metrics["supporting_fact_exact_match"] == 0.0
        assert serialized_metrics["joint_exact_match"] == 0.0
        assert serialized_metrics["supporting_fact_prediction_coverage"] == 0.0
        assert serialized_metrics["official_hotpotqa"]["em"] == 0.5
        assert serialized_metrics["official_hotpotqa"]["sp_f1"] == 0.0
        assert serialized_metrics["official_hotpotqa"]["joint_f1"] == 0.0
        assert serialized_metrics["correct"] == 1
        assert serialized_metrics["incorrect"] == 1
        assert serialized_metrics["average_steps"] == 1.0
        assert serialized_metrics["average_tool_calls"] == 0.0
        assert serialized_metrics["termination_reasons"] == {"mock_completed": 2}
        assert len(prediction_lines) == 2
        assert json.loads(prediction_lines[0])["correct"] is True
        assert json.loads(prediction_lines[0])["gold_supporting_facts"] == [
            ["Article A", 0]
        ]
        log_text = log_stream.getvalue()
        assert "[1/2] example_id=example-1" in log_text
        assert "Question: First question?" in log_text
        assert "[2/2] example_id=example-2" in log_text
        assert "Question: Second question?" in log_text
        assert log_text.index("Question: First question?") < log_text.index(
            "Prediction: answer"
        )
        assert "Prediction: <EMPTY>" in log_text
        assert "Termination: mock_completed" in log_text
        assert "=== FINAL HOTPOTQA METRICS ===" in log_text
        assert "metric.em=0.500000" in log_text
        assert "metric.sp_f1=0.000000" in log_text
        assert "metric.joint_f1=0.000000" in log_text
        assert result.artifacts.run_directory.name == "2026-08-23_013000"
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        if test_output_root.exists():
            resolved_test_root = test_output_root.resolve()
            assert resolved_test_root.parent == (PROJECT_ROOT / "outputs").resolve()
            shutil.rmtree(resolved_test_root)


def test_react_benchmark_flushes_one_trajectory_per_example() -> None:
    test_output_root = PROJECT_ROOT / "outputs" / f"_pytest_{uuid4().hex}"
    examples = [BenchmarkExample("example-1", "When?", "1879")]
    config = BenchmarkRunConfig(
        model="scripted",
        dataset="hotpotqa/hotpot_qa:distractor:validation",
        task="hotpotqa",
        method="react",
        num_samples=1,
        seed=42,
        generation={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 16},
        max_agent_steps=3,
        timestamp="2026-08-23T01:31:00+00:00",
    )

    class TrajectoryPredictor:
        def predict(self, example: BenchmarkExample) -> AgentResult:
            return AgentResult(
                prediction="1879",
                steps=1,
                tool_calls=0,
                termination_reason="completed",
                trajectory=(
                    TrajectoryStep(
                        step_index=1,
                        model_output="Thought: done\nAction: Finish[1879]",
                        thought="done",
                        action="Finish[1879]",
                        observation="Finished with answer: 1879",
                    ),
                ),
            )

    try:
        result = run_hotpotqa_benchmark(
            examples,
            TrajectoryPredictor(),
            config,
            output_root=test_output_root,
        )
        lines = result.artifacts.trajectories_path.read_text("utf-8").splitlines()
        assert len(lines) == 1
        serialized = json.loads(lines[0])
        assert serialized["method"] == "react"
        assert serialized["termination_reason"] == "completed"
        assert serialized["steps"][0]["thought"] == "done"
        assert serialized["steps"][0]["action"] == "Finish[1879]"
    finally:
        if test_output_root.exists():
            resolved_test_root = test_output_root.resolve()
            assert resolved_test_root.parent == (PROJECT_ROOT / "outputs").resolve()
            shutil.rmtree(resolved_test_root)
