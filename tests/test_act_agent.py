"""Integration-style Act-only tests using scripted model output."""

from __future__ import annotations

from react_reproduction.agents.act import ActOnlyAgent
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.tools.wikipedia import WikipediaEnvironment
from tests.test_wikipedia import FakeWikipediaClient


class ScriptedLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        return next(self._responses)


def test_act_agent_searches_looks_up_and_finishes() -> None:
    agent = ActOnlyAgent(
        ScriptedLLM(
            [
                "Action: Search[Albert Einstein]",
                "Action: Lookup[born]",
                "Action: Finish[1879]",
            ]
        ),
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
    )

    result = agent.predict(BenchmarkExample("1", "When was Einstein born?", "1879"))

    assert result.prediction == "1879"
    assert result.steps == 3
    assert result.tool_calls == 2
    assert result.termination_reason == "completed"
    assert all(step.thought is None for step in result.trajectory)


def test_act_agent_recovers_from_one_parsing_error() -> None:
    agent = ActOnlyAgent(
        ScriptedLLM(["I need a search", "Action: Finish[1879]"]),
        GenerationConfig(max_new_tokens=32),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=3),
        max_steps=3,
    )
    result = agent.predict(BenchmarkExample("1", "When?", "1879"))
    assert result.prediction == "1879"
    assert result.steps == 2
    assert "Invalid action format" in (result.trajectory[0].observation or "")
