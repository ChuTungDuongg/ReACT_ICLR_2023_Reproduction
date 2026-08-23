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


class StaticAgent(BaseAgent):
    def __init__(self, result: AgentResult) -> None:
        self.result = result
        self.calls = 0

    def predict(self, example: BenchmarkExample) -> AgentResult:
        self.calls += 1
        return self.result


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
