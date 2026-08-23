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


class BatchedScriptedLLM(LLMProvider):
    def __init__(self, response_batches: list[list[str]]) -> None:
        self._response_batches = iter(response_batches)
        self.prompt_batches: list[list[str]] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        raise AssertionError("Batched Act must use generate_batch().")

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        self.prompt_batches.append(list(prompts))
        responses = next(self._response_batches)
        assert len(responses) == len(prompts)
        return tuple(responses)


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


def test_act_agent_uses_final_step_for_best_effort_answer() -> None:
    llm = ScriptedLLM(
        [
            "Action: Search[Albert Einstein]",
            "Action: Finish[1879]",
        ]
    )
    agent = ActOnlyAgent(
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
    assert "Action 2: Finish[<best answer>]" in llm.prompts[1]


def test_act_batches_questions_with_independent_wikipedia_state() -> None:
    llm = BatchedScriptedLLM(
        [
            ["Action: Search[Albert Einstein]", "Action: Search[missing]"],
            ["Action: Finish[1879]", "Action: Finish[unknown]"],
        ]
    )
    created_environments: list[WikipediaEnvironment] = []

    def create_environment() -> WikipediaEnvironment:
        environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=3)
        created_environments.append(environment)
        return environment

    agent = ActOnlyAgent(
        llm,
        GenerationConfig(max_new_tokens=32),
        create_environment(),
        max_steps=3,
        environment_factory=create_environment,
    )
    examples = [
        BenchmarkExample("1", "When was Einstein born?", "1879"),
        BenchmarkExample("2", "What is missing?", "unknown"),
    ]

    results = agent.predict_batch(examples)

    assert [result.prediction for result in results] == ["1879", "unknown"]
    assert [len(batch) for batch in llm.prompt_batches] == [2, 2]
    assert "Opened 'Albert Einstein'" in llm.prompt_batches[1][0]
    assert "No Wikipedia article found" in llm.prompt_batches[1][1]
    assert len(created_environments) == 2
