"""Model-free tests for Standard and Chain-of-Thought agents."""

from __future__ import annotations

from react_reproduction.agents.cot import CoTAgent
from react_reproduction.agents.standard import StandardAgent
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider


class StubLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self.response = response
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
        return self.response


def test_standard_agent_returns_parsed_answer_without_tools() -> None:
    llm = StubLLM("Final Answer: Berlin")
    agent = StandardAgent(llm, GenerationConfig(max_new_tokens=32))

    result = agent.predict(BenchmarkExample("1", "Capital of Germany?", "Berlin"))

    assert result.prediction == "Berlin"
    assert result.steps == 1
    assert result.tool_calls == 0
    assert result.termination_reason == "completed"
    assert "six manually composed examples" in llm.prompts[0]
    assert llm.prompts[0].count("Question:") == 7


def test_cot_agent_records_reasoning_in_trajectory() -> None:
    llm = StubLLM("Reasoning: Germany's capital is Berlin.\nFinal Answer: Berlin")
    agent = CoTAgent(llm, GenerationConfig(max_new_tokens=64))

    result = agent.predict(BenchmarkExample("1", "Capital of Germany?", "Berlin"))

    assert result.prediction == "Berlin"
    assert result.trajectory[0].thought == "Germany's capital is Berlin."
    assert result.tool_calls == 0


def test_cot_agent_parses_paper_style_thought_and_answer() -> None:
    llm = StubLLM("Thought: Germany's capital is Berlin.\nAnswer: Berlin")
    agent = CoTAgent(llm, GenerationConfig(max_new_tokens=64))

    result = agent.predict(BenchmarkExample("1", "Capital of Germany?", "Berlin"))

    assert result.prediction == "Berlin"
    assert result.trajectory[0].thought == "Germany's capital is Berlin."
