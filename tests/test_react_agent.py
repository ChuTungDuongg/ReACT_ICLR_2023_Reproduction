"""Integration-style ReAct tests using scripted model outputs."""

from __future__ import annotations

from react_reproduction.agents.react import ReActAgent
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.tools.wikipedia import WikipediaEnvironment
from tests.test_wikipedia import FakeWikipediaClient


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        self.prompts.append(prompt)
        return next(self._responses)


def test_react_agent_reasons_searches_looks_up_and_finishes() -> None:
    llm = ScriptedLLM(
        [
            "Thought: Find Einstein.\nAction: Search[Albert Einstein]",
            "Thought: Find his birth sentence.\nAction: Lookup[born]",
            "Thought: The page gives 1879.\nAction: Finish[1879]",
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
    )

    result = agent.predict(BenchmarkExample("1", "When was Einstein born?", "1879"))

    assert result.prediction == "1879"
    assert result.steps == 3
    assert result.tool_calls == 2
    assert result.termination_reason == "completed"
    assert result.trajectory[0].thought == "Find Einstein."
    assert "Observation 1:" in llm.prompts[1]


def test_react_agent_recovers_from_one_parsing_error() -> None:
    agent = ReActAgent(
        ScriptedLLM(
            [
                "Thought: I need evidence, but no action follows.",
                "Thought: I can now answer.\nAction: Finish[1879]",
            ]
        ),
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=3),
        max_steps=3,
    )

    result = agent.predict(BenchmarkExample("1", "When?", "1879"))

    assert result.prediction == "1879"
    assert result.steps == 2
    assert result.trajectory[0].thought == "I need evidence, but no action follows."
    assert "Invalid action format" in (result.trajectory[0].observation or "")


def test_react_agent_rejects_tool_action_on_forced_final_step() -> None:
    llm = ScriptedLLM(
        [
            "Thought: Search once.\nAction: Search[Albert Einstein]",
            "Thought: Search again.\nAction: Search[Albert Einstein]",
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=2),
        max_steps=2,
    )

    result = agent.predict(BenchmarkExample("1", "When?", "1879"))

    assert result.prediction == ""
    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.termination_reason == "finalization_failed"
    assert "final allowed step" in llm.prompts[1]
    assert "tool action was not executed" in (result.trajectory[1].observation or "")


def test_react_agent_uses_final_step_for_best_effort_answer() -> None:
    llm = ScriptedLLM(
        [
            "Thought: Find Einstein.\nAction: Search[Albert Einstein]",
            "Thought: The evidence says 1879.\nAction: Finish[1879]",
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=2),
        max_steps=2,
    )

    result = agent.predict(BenchmarkExample("1", "When?", "1879"))

    assert result.prediction == "1879"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.termination_reason == "completed_at_step_limit"
    assert "Action 2: Finish[<best concise answer>]" in llm.prompts[1]


def test_react_agent_forces_finish_after_action_loop() -> None:
    llm = ScriptedLLM(
        [
            "Thought: Search.\nAction: Search[Albert Einstein]",
            "Thought: Repeat.\nAction: Search[Albert Einstein]",
            "Thought: Repeat again.\nAction: Search[Albert Einstein]",
            "Thought: Give the best answer.\nAction: Finish[1879]",
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
    )

    result = agent.predict(BenchmarkExample("1", "When?", "1879"))

    assert result.prediction == "1879"
    assert result.steps == 4
    assert result.tool_calls == 1
    assert result.termination_reason == "completed_after_action_loop"
    assert "Repeated Search ignored" in (result.trajectory[1].observation or "")
    assert "final allowed step" in llm.prompts[3]


def test_react_agent_can_disable_best_effort_for_paper_hybrid_fallback() -> None:
    llm = ScriptedLLM(
        [
            "Thought: Search.\nAction: Search[Albert Einstein]",
            "Thought: Search again.\nAction: Search[Albert Einstein]",
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=2),
        max_steps=2,
        best_effort_finalization=False,
    )

    result = agent.predict(BenchmarkExample("1", "When?", "1879"))

    assert result.prediction == ""
    assert result.termination_reason == "max_steps_exceeded"
    assert "final allowed step" not in llm.prompts[1]
