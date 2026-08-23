"""Offline tests for CoT-SC and both paper-style hybrid fallback orders."""

from __future__ import annotations

import pytest

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.cot_sc import CoTSCAgent
from react_reproduction.agents.hybrid import CoTSCThenReActAgent, ReActThenCoTSCAgent
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider


EXAMPLE = BenchmarkExample("example-1", "What is the answer?", "Berlin")
EXAMPLE_2 = BenchmarkExample("example-2", "What is the second answer?", "Rome")


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.temperatures: list[float] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        self.temperatures.append(temperature)
        return next(self._responses)


class BatchedScriptedLLM(LLMProvider):
    def __init__(self, response_batches: list[list[str]]) -> None:
        self._response_batches = iter(response_batches)
        self.batch_sizes: list[int] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        raise AssertionError("The batched CoT-SC path must use generate_batch().")

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        self.batch_sizes.append(len(prompts))
        responses = next(self._response_batches)
        assert len(responses) == len(prompts)
        return tuple(responses)


class StaticAgent(BaseAgent):
    def __init__(self, result: AgentResult) -> None:
        self.result = result
        self.calls = 0

    def predict(self, example: BenchmarkExample) -> AgentResult:
        self.calls += 1
        return self.result


class BatchStaticAgent(BaseAgent):
    def __init__(self, results_by_id: dict[str, AgentResult]) -> None:
        self.results_by_id = results_by_id
        self.batch_sizes: list[int] = []

    def predict(self, example: BenchmarkExample) -> AgentResult:
        return self.results_by_id[example.example_id]

    def predict_batch(
        self,
        examples: list[BenchmarkExample],
    ) -> tuple[AgentResult, ...]:
        self.batch_sizes.append(len(examples))
        return tuple(self.predict(example) for example in examples)


def _cot_sc(responses: list[str]) -> CoTSCAgent:
    return CoTSCAgent(
        ScriptedLLM(responses),
        GenerationConfig(temperature=0.7, max_new_tokens=32),
        num_samples=len(responses),
    )


def _react_result(
    prediction: str,
    termination_reason: str,
    *,
    tool_calls: int = 1,
) -> AgentResult:
    return AgentResult(
        prediction=prediction,
        steps=1,
        tool_calls=tool_calls,
        termination_reason=termination_reason,
        trajectory=(
            TrajectoryStep(
                step_index=1,
                model_output=f"Action: Finish[{prediction}]" if prediction else "bad",
                action=f"Finish[{prediction}]" if prediction else None,
            ),
        ),
    )


def test_cot_sc_requires_positive_sampling_temperature() -> None:
    with pytest.raises(ValueError, match="positive sampling temperature"):
        CoTSCAgent(
            ScriptedLLM(["Answer: Berlin"]),
            GenerationConfig(temperature=0.0),
            num_samples=1,
        )


def test_cot_sc_normalizes_votes_and_uses_sampling_temperature() -> None:
    llm = ScriptedLLM(
        [
            "Thought: one\nAnswer: The Berlin",
            "Thought: two\nAnswer: berlin!",
            "Thought: three\nAnswer: Munich",
            "Thought: four\nAnswer: Berlin",
            "Thought: five\nAnswer: Hamburg",
        ]
    )
    agent = CoTSCAgent(
        llm,
        GenerationConfig(temperature=0.7, max_new_tokens=32),
        num_samples=5,
    )

    outcome = agent.predict_with_consensus(EXAMPLE)

    assert outcome.result.prediction == "The Berlin"
    assert outcome.winning_count == 3
    assert outcome.confidence == 0.6
    assert outcome.meets_paper_threshold is True
    assert outcome.result.metadata["cot_sc_valid_answers"] == 5
    assert outcome.result.metadata["cot_sc_vote_counts"] == {
        "berlin": 3,
        "munich": 1,
        "hamburg": 1,
    }
    assert llm.temperatures == [0.7] * 5


def test_cot_sc_rejects_malformed_votes_and_detects_low_confidence() -> None:
    agent = _cot_sc(
        [
            "Thought only, with no labeled answer",
            "Answer: Berlin",
            "Answer: Berlin",
            "Answer: Munich",
            "Answer: Hamburg",
        ]
    )

    outcome = agent.predict_with_consensus(EXAMPLE)

    assert outcome.valid_answer_count == 4
    assert outcome.winning_count == 2
    assert outcome.meets_paper_threshold is False


def test_cot_sc_accepts_exact_half_for_even_sample_count_per_paper_rule() -> None:
    outcome = _cot_sc(
        ["Answer: Berlin", "Answer: Berlin", "Answer: Paris", "Answer: Rome"]
    ).predict_with_consensus(EXAMPLE)

    assert outcome.winning_count == 2
    assert outcome.meets_paper_threshold is True


def test_cot_sc_batches_multiple_questions_for_every_sampling_round() -> None:
    llm = BatchedScriptedLLM(
        [
            ["Answer: Berlin", "Answer: Paris"],
            ["Answer: Berlin", "Answer: Rome"],
            ["Answer: Munich", "Answer: Rome"],
        ]
    )
    agent = CoTSCAgent(
        llm,
        GenerationConfig(temperature=0.7, max_new_tokens=32),
        num_samples=3,
    )
    examples = [
        EXAMPLE,
        BenchmarkExample("example-2", "Second question?", "Rome"),
    ]

    results = agent.predict_batch(examples)

    assert [result.prediction for result in results] == ["Berlin", "Rome"]
    assert [result.steps for result in results] == [3, 3]
    assert llm.batch_sizes == [2, 2, 2]


def test_react_then_cot_sc_keeps_natural_react_answer_without_fallback() -> None:
    react = StaticAgent(_react_result("Berlin", "completed"))
    cot_sc = _cot_sc(["Answer: Paris"])
    agent = ReActThenCoTSCAgent(react, cot_sc)

    result = agent.predict(EXAMPLE)

    assert result.prediction == "Berlin"
    assert result.termination_reason == "react_completed"
    assert result.metadata["fallback_used"] is False
    assert result.trajectory[0].phase == "react"


def test_react_then_cot_sc_falls_back_after_react_failure() -> None:
    react = StaticAgent(_react_result("", "action_loop", tool_calls=3))
    agent = ReActThenCoTSCAgent(
        react,
        _cot_sc(["Answer: Berlin", "Answer: Berlin", "Answer: Paris"]),
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "Berlin"
    assert result.termination_reason == "fallback_cot_sc"
    assert result.tool_calls == 3
    assert result.metadata["fallback_used"] is True
    assert result.metadata["react_termination_reason"] == "action_loop"
    assert [step.phase for step in result.trajectory] == [
        "react",
        "cot_sc",
        "cot_sc",
        "cot_sc",
    ]
    assert [step.step_index for step in result.trajectory] == [1, 2, 3, 4]


def test_cot_sc_then_react_keeps_confident_consensus_without_fallback() -> None:
    react = StaticAgent(_react_result("Paris", "completed"))
    agent = CoTSCThenReActAgent(
        _cot_sc(["Answer: Berlin", "Answer: Berlin", "Answer: Paris"]),
        react,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "Berlin"
    assert result.termination_reason == "cot_sc_consensus"
    assert result.metadata["fallback_used"] is False
    assert react.calls == 0


def test_cot_sc_then_react_falls_back_on_low_confidence() -> None:
    react = StaticAgent(_react_result("Berlin", "completed", tool_calls=2))
    agent = CoTSCThenReActAgent(
        _cot_sc(
            [
                "Answer: Berlin",
                "Answer: Berlin",
                "Answer: Paris",
                "Answer: Paris",
                "Answer: Rome",
            ]
        ),
        react,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "Berlin"
    assert result.termination_reason == "fallback_react_completed"
    assert result.tool_calls == 2
    assert result.metadata["selected_path"] == "react"
    assert result.metadata["cot_sc_threshold_met"] is False
    assert react.calls == 1
    assert result.trajectory[-1].phase == "react"


def test_react_then_cot_sc_batches_both_legs_and_preserves_order() -> None:
    react = BatchStaticAgent(
        {
            EXAMPLE.example_id: _react_result("", "max_steps_exceeded"),
            EXAMPLE_2.example_id: _react_result("", "action_loop"),
        }
    )
    llm = BatchedScriptedLLM(
        [
            ["Answer: Berlin", "Answer: Rome"],
            ["Answer: Berlin", "Answer: Rome"],
            ["Answer: Paris", "Answer: Paris"],
        ]
    )
    cot_sc = CoTSCAgent(
        llm,
        GenerationConfig(temperature=0.7, max_new_tokens=32),
        num_samples=3,
    )

    results = ReActThenCoTSCAgent(react, cot_sc).predict_batch(
        [EXAMPLE, EXAMPLE_2]
    )

    assert [result.prediction for result in results] == ["Berlin", "Rome"]
    assert react.batch_sizes == [2]
    assert llm.batch_sizes == [2, 2, 2]
    assert all(result.metadata["selected_path"] == "cot_sc" for result in results)


def test_cot_sc_then_react_batches_low_confidence_fallbacks() -> None:
    llm = BatchedScriptedLLM(
        [
            ["Answer: Paris", "Answer: Madrid"],
            ["Answer: Munich", "Answer: Lisbon"],
            ["Answer: Hamburg", "Answer: Vienna"],
        ]
    )
    cot_sc = CoTSCAgent(
        llm,
        GenerationConfig(temperature=0.7, max_new_tokens=32),
        num_samples=3,
    )
    react = BatchStaticAgent(
        {
            EXAMPLE.example_id: _react_result("Berlin", "completed"),
            EXAMPLE_2.example_id: _react_result("Rome", "completed"),
        }
    )

    results = CoTSCThenReActAgent(cot_sc, react).predict_batch(
        [EXAMPLE, EXAMPLE_2]
    )

    assert [result.prediction for result in results] == ["Berlin", "Rome"]
    assert llm.batch_sizes == [2, 2, 2]
    assert react.batch_sizes == [2]
    assert all(result.metadata["selected_path"] == "react" for result in results)
