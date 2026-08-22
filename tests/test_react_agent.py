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


def test_react_agent_stops_at_max_steps() -> None:
    agent = ReActAgent(
        ScriptedLLM(
            [
                "Thought: Search once.\nAction: Search[Albert Einstein]",
                "Thought: Search again.\nAction: Search[Albert Einstein]",
            ]
        ),
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=2),
        max_steps=2,
    )

    result = agent.predict(BenchmarkExample("1", "When?", "1879"))

    assert result.prediction == ""
    assert result.steps == 2
    assert result.termination_reason == "max_steps_exceeded"
