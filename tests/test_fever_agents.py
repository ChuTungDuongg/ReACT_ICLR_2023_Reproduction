"""Model-free tests for all FEVER agent paths and paper heuristics."""

from __future__ import annotations

import pytest

from react_reproduction.agents.act import ActOnlyAgent
from react_reproduction.agents.base import AgentResult, BaseAgent
from react_reproduction.agents.cot import CoTAgent
from react_reproduction.agents.cot_sc import CoTSCAgent
from react_reproduction.agents.hybrid import CoTSCThenReActAgent, ReActThenCoTSCAgent
from react_reproduction.agents.react import ReActAgent
from react_reproduction.agents.standard import StandardAgent
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.fever import normalize_fever_label, parse_fever_label
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.fever import (
    build_act_prompt,
    build_cot_prompt,
    build_react_prompt,
    build_standard_prompt,
)
from react_reproduction.tools.wikipedia import WikipediaEnvironment
from tests.test_wikipedia import FakeWikipediaClient


EXAMPLE = BenchmarkExample("fever-1", "A verifiable claim.", "SUPPORTS")


class BatchLLM(LLMProvider):
    def __init__(self, batches: list[list[str]]) -> None:
        self._batches = iter(batches)
        self.temperatures: list[float] = []
        self.batch_sizes: list[int] = []
        self.prompt_batches: list[list[str]] = []
        self.stop_sequences: list[tuple[str, ...] | None] = []

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop_sequences: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        return self.generate_batch(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )[0]

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop_sequences: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        self.temperatures.append(temperature)
        self.batch_sizes.append(len(prompts))
        self.prompt_batches.append(list(prompts))
        normalized_stops = (
            tuple(stop_sequences) if stop_sequences is not None else None
        )
        self.stop_sequences.append(normalized_stops)
        result = next(self._batches)
        assert len(result) == len(prompts)
        if not normalized_stops:
            return tuple(result)
        truncated: list[str] = []
        for output in result:
            stop_indexes = [
                output.find(stop)
                for stop in normalized_stops
                if output.find(stop) >= 0
            ]
            truncated.append(output[: min(stop_indexes)] if stop_indexes else output)
        return tuple(truncated)


class StaticAgent(BaseAgent):
    def __init__(self, result: AgentResult) -> None:
        self.result = result
        self.calls = 0

    def predict(self, example: BenchmarkExample) -> AgentResult:
        self.calls += 1
        return self.result


def _fever_cot_sc(outputs: list[str]) -> CoTSCAgent:
    return CoTSCAgent(
        BatchLLM([[output] for output in outputs]),
        GenerationConfig(temperature=0.7, max_new_tokens=32),
        num_samples=len(outputs),
        prompt_builder=build_cot_prompt,
        answer_parser=parse_fever_label,
        answer_normalizer=normalize_fever_label,
    )


def test_standard_and_cot_reuse_shared_agents_with_fever_adapters() -> None:
    standard = StandardAgent(
        BatchLLM([["Answer: The claim is supported."]]),
        GenerationConfig(),
        prompt_builder=build_standard_prompt,
        answer_parser=parse_fever_label,
        invalid_termination_reason="invalid_label",
    )
    cot = CoTAgent(
        BatchLLM([["Thought: evidence contradicts it.\nAnswer: refuted"]]),
        GenerationConfig(),
        prompt_builder=build_cot_prompt,
        answer_parser=parse_fever_label,
        invalid_termination_reason="invalid_label",
    )

    assert standard.predict(EXAMPLE).prediction == "SUPPORTS"
    assert cot.predict(EXAMPLE).prediction == "REFUTES"


def test_fever_standard_stops_after_the_first_answer_line() -> None:
    llm = BatchLLM([["REFUTES\nThis explanation should not be generated."]])
    agent = StandardAgent(
        llm,
        GenerationConfig(),
        prompt_builder=build_standard_prompt,
        answer_parser=parse_fever_label,
        invalid_termination_reason="invalid_label",
        stop_sequences=("\n",),
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "REFUTES"
    assert result.trajectory[0].model_output == "REFUTES"
    assert llm.stop_sequences == [("\n",)]


@pytest.mark.parametrize("label", ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
@pytest.mark.parametrize("agent_type", ["act", "react"])
def test_interactive_agents_normalize_all_finish_labels(
    label: str,
    agent_type: str,
) -> None:
    output = (
        f"Action 1: Finish[{label}]"
        if agent_type == "act"
        else f"Thought 1: Evidence is sufficient.\nAction 1: Finish[{label}]"
    )
    kwargs = {
        "llm": BatchLLM([[output]]),
        "generation": GenerationConfig(),
        "environment": WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        "max_steps": 5,
        "answer_normalizer": normalize_fever_label,
    }
    agent = (
        ActOnlyAgent(prompt_builder=build_act_prompt, **kwargs)
        if agent_type == "act"
        else ReActAgent(prompt_builder=build_react_prompt, **kwargs)
    )
    assert agent.predict(EXAMPLE).prediction == label


def test_invalid_finish_label_is_not_fuzzy_mapped() -> None:
    agent = ReActAgent(
        BatchLLM([["Thought 1: uncertain.\nAction 1: Finish[probably supports]"]]),
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )
    result = agent.predict(EXAMPLE)
    assert result.prediction == ""
    assert result.termination_reason == "invalid_label"
    assert result.metadata["raw_prediction"] == "probably supports"


def test_cot_sc_generates_exactly_21_samples_at_temperature_point_7() -> None:
    outputs = ["Thought: yes.\nAnswer: SUPPORTS"] * 11 + [
        "Thought: no.\nAnswer: REFUTES"
    ] * 10
    agent = _fever_cot_sc(outputs)
    outcome = agent.predict_with_consensus(EXAMPLE)

    assert outcome.sample_count == 21
    assert outcome.winning_count == 11
    assert outcome.meets_paper_threshold is True
    assert len(outcome.result.trajectory) == 21
    assert len(outcome.result.metadata["cot_sc_generated_samples"]) == 21
    assert len(outcome.result.metadata["cot_sc_normalized_labels"]) == 21
    assert outcome.result.metadata["vote_distribution"] == {
        "SUPPORTS": 11,
        "REFUTES": 10,
    }
    assert outcome.result.termination_reason == "cot_sc_majority"
    assert agent._llm.temperatures == [0.7] * 21


def test_cot_sc_counts_safe_fever_variants_and_discards_ambiguous_votes() -> None:
    outputs = [
        "Answer: REFUTES",
        "Answer: REFUTES.",
        "Answer: REFUTES (the evidence contradicts the claim)",
        "Answer: SUPPORTS (the evidence confirms the claim)",
        "Answer: NOT ENOUGH INFO (the evidence is insufficient)",
        "Answer: REFUTES or NOT ENOUGH INFO",
        "Answer: SUPPORTS / REFUTES",
    ]

    outcome = _fever_cot_sc(outputs).predict_with_consensus(EXAMPLE)

    assert outcome.valid_answer_count == 5
    assert outcome.winning_count == 3
    assert outcome.result.prediction == "REFUTES"
    assert outcome.result.metadata["vote_distribution"] == {
        "REFUTES": 3,
        "SUPPORTS": 1,
        "NOT ENOUGH INFO": 1,
    }
    assert outcome.result.metadata["cot_sc_normalized_labels"] == [
        "REFUTES",
        "REFUTES",
        "REFUTES",
        "SUPPORTS",
        "NOT ENOUGH INFO",
        "",
        "",
    ]


def test_cot_sc_low_vote_winner_is_recorded_as_plurality() -> None:
    outcome = _fever_cot_sc(
        ["Answer: SUPPORTS"] * 10
        + ["Answer: REFUTES"] * 9
        + ["Answer: NOT ENOUGH INFO"] * 2
    ).predict_with_consensus(EXAMPLE)
    assert outcome.winning_count == 10
    assert outcome.meets_paper_threshold is False
    assert outcome.result.termination_reason == "cot_sc_plurality"
    assert outcome.result.metadata["paper_threshold_met"] is False


def test_cot_sc_then_react_uses_11_of_21_without_fallback() -> None:
    react = StaticAgent(AgentResult("REFUTES", 1, 0, "completed"))
    result = CoTSCThenReActAgent(
        _fever_cot_sc(["Answer: SUPPORTS"] * 11 + ["Answer: REFUTES"] * 10),
        react,
    ).predict(EXAMPLE)
    assert result.prediction == "SUPPORTS"
    assert result.termination_reason == "cot_sc_consensus"
    assert result.metadata["execution_path"] == "cot_sc_consensus"
    assert react.calls == 0


def test_cot_sc_then_react_falls_back_at_10_of_21() -> None:
    react = StaticAgent(AgentResult("REFUTES", 1, 1, "completed"))
    result = CoTSCThenReActAgent(
        _fever_cot_sc(
            ["Answer: SUPPORTS"] * 10
            + ["Answer: REFUTES"] * 9
            + ["Answer: NOT ENOUGH INFO"] * 2
        ),
        react,
    ).predict(EXAMPLE)
    assert result.prediction == "REFUTES"
    assert result.termination_reason == "fallback_react_completed"
    assert result.metadata["execution_path"] == "fallback_react_completed"
    assert react.calls == 1


def test_react_then_cot_sc_falls_back_only_after_react_failure() -> None:
    completed = StaticAgent(AgentResult("SUPPORTS", 1, 0, "completed"))
    cot_sc = _fever_cot_sc(["Answer: REFUTES"] * 21)
    kept = ReActThenCoTSCAgent(completed, cot_sc).predict(EXAMPLE)
    assert kept.prediction == "SUPPORTS"
    assert kept.metadata["execution_path"] == "react_completed"

    failed = StaticAgent(AgentResult("", 5, 4, "max_steps_exceeded"))
    fallback = ReActThenCoTSCAgent(
        failed,
        _fever_cot_sc(["Answer: REFUTES"] * 21),
    ).predict(EXAMPLE)
    assert fallback.prediction == "REFUTES"
    assert fallback.termination_reason == "fallback_cot_sc"
    assert fallback.metadata["execution_path"] == "fallback_cot_sc"


def test_fever_react_batch_state_is_isolated() -> None:
    llm = BatchLLM(
        [
            [
                "Thought 1: Search.\nAction 1: Search[Albert Einstein]",
                "Thought 1: Search.\nAction 1: Search[missing]",
            ],
            [
                "Thought 2: Supported.\nAction 2: Finish[SUPPORTS]",
                "Thought 2: Unknown.\nAction 2: Finish[NOT ENOUGH INFO]",
            ],
        ]
    )
    created: list[WikipediaEnvironment] = []

    def factory() -> WikipediaEnvironment:
        environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=5)
        created.append(environment)
        return environment

    agent = ReActAgent(
        llm,
        GenerationConfig(),
        factory(),
        max_steps=5,
        environment_factory=factory,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )
    results = agent.predict_batch(
        [EXAMPLE, BenchmarkExample("fever-2", "Another claim.", "NOT ENOUGH INFO")]
    )
    assert [result.prediction for result in results] == [
        "SUPPORTS",
        "NOT ENOUGH INFO",
    ]
    assert "Opened 'Albert Einstein'" in llm.prompt_batches[1][0]
    assert "Could not find [missing]" in llm.prompt_batches[1][1]
    assert len(created) == 2


def test_fever_react_stops_before_model_generated_observation() -> None:
    llm = BatchLLM(
        [
            [
                "Thought 1: Search for the entity.\n"
                "Action 1: Search[Albert Einstein]\n"
                "Observation 1: FABRICATED MODEL OBSERVATION\n"
                "Thought 2: Finish immediately.\n"
                "Action 2: Finish[REFUTES]"
            ],
            ["Thought 2: The real evidence supports it.\nAction 2: Finish[SUPPORTS]"],
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "SUPPORTS"
    assert result.tool_calls == 1
    assert result.trajectory[0].model_output.endswith(
        "Action 1: Search[Albert Einstein]"
    )
    assert "FABRICATED MODEL OBSERVATION" not in llm.prompt_batches[1][0]
    assert "Opened 'Albert Einstein'" in llm.prompt_batches[1][0]
    assert llm.stop_sequences == [
        ("\nObservation 1:",),
        ("\nObservation 2:",),
    ]


def test_fever_react_recovers_malformed_action_within_the_same_step() -> None:
    llm = BatchLLM(
        [
            [
                "Thought 1: The evidence is sufficient.\n"
                "Action 1: Therefore I should reject the claim."
            ],
            ["Finish[REFUTES]\nObservation 1: FABRICATED MODEL OBSERVATION"],
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "REFUTES"
    assert result.termination_reason == "completed"
    assert result.steps == 1
    assert result.tool_calls == 0
    assert len(result.trajectory) == 1
    assert result.trajectory[0].thought == "The evidence is sufficient."
    assert result.trajectory[0].action == "Finish[REFUTES]"
    assert "FABRICATED MODEL OBSERVATION" not in result.trajectory[0].model_output
    assert llm.prompt_batches[1][0].endswith(
        "Thought 1: The evidence is sufficient.\nAction 1:"
    )
    assert llm.stop_sequences == [("\nObservation 1:",), ("\n",)]


def test_fever_react_recovery_uses_unlabeled_first_reasoning_line() -> None:
    llm = BatchLLM(
        [
            ["The evidence is sufficient.\nAction 1: This is malformed prose."],
            ["Finish[REFUTES]"],
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "REFUTES"
    assert result.steps == 1
    assert result.trajectory[0].thought == "The evidence is sufficient."
    assert llm.prompt_batches[1][0].endswith(
        "Thought 1: The evidence is sufficient.\nAction 1:"
    )


def test_fever_react_recovered_search_feeds_real_observation_to_next_step() -> None:
    llm = BatchLLM(
        [
            [
                "Thought 1: I should verify the university.\n"
                "Action 1: I will search for the university."
            ],
            ["Search[Peking University]"],
            ["Thought 2: It was founded in 1898.\nAction 2: Finish[REFUTES]"],
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "REFUTES"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert len(result.trajectory) == 2
    assert result.trajectory[0].action == "Search[Peking University]"
    assert "Opened top result 'Albert Einstein'" in llm.prompt_batches[2][0]
    assert "I will search for the university" not in llm.prompt_batches[2][0]
    assert llm.stop_sequences == [
        ("\nObservation 1:",),
        ("\n",),
        ("\nObservation 2:",),
    ]


def test_fever_react_failed_recovery_consumes_one_logical_step_only() -> None:
    llm = BatchLLM(
        [
            ["Thought 1: I need evidence.\nAction 1: This is malformed prose."],
            ["This is still not a tool action.\nThought 2: Do not continue."],
        ]
    )
    agent = ReActAgent(
        llm,
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=1,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == ""
    assert result.termination_reason == "parsing_error"
    assert result.steps == 1
    assert result.tool_calls == 0
    assert len(result.trajectory) == 1
    assert result.trajectory[0].action is None
    assert "Invalid action format" in (result.trajectory[0].observation or "")
    assert llm.batch_sizes == [1, 1]
    assert llm.stop_sequences == [("\nObservation 1:",), ("\n",)]


def test_fever_react_mixed_batch_recovers_only_the_failed_state() -> None:
    llm = BatchLLM(
        [
            [
                "Thought 1: Search normally.\nAction 1: Search[Albert Einstein]",
                "Thought 1: The claim is contradicted.\n"
                "Action 1: Therefore I should reject it.",
            ],
            ["Finish[REFUTES]"],
            ["Thought 2: The page is sufficient.\nAction 2: Finish[SUPPORTS]"],
        ]
    )
    created: list[WikipediaEnvironment] = []

    def factory() -> WikipediaEnvironment:
        environment = WikipediaEnvironment(FakeWikipediaClient(), max_steps=5)
        created.append(environment)
        return environment

    agent = ReActAgent(
        llm,
        GenerationConfig(),
        factory(),
        max_steps=5,
        environment_factory=factory,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )
    examples = [
        EXAMPLE,
        BenchmarkExample("fever-2", "A contradicted claim.", "REFUTES"),
    ]

    results = agent.predict_batch(examples)

    assert [result.prediction for result in results] == ["SUPPORTS", "REFUTES"]
    assert [result.steps for result in results] == [2, 1]
    assert [result.tool_calls for result in results] == [1, 0]
    assert llm.batch_sizes == [2, 1, 1]
    assert "A contradicted claim." in llm.prompt_batches[1][0]
    assert "A verifiable claim." not in llm.prompt_batches[1][0]
    assert llm.prompt_batches[1][0].endswith(
        "Thought 1: The claim is contradicted.\nAction 1:"
    )
    assert "Opened 'Albert Einstein'" in llm.prompt_batches[2][0]
    assert len(created) == 2


def test_fever_act_generation_executes_only_the_current_action() -> None:
    llm = BatchLLM(
        [
            [
                "Action 1: Search[Albert Einstein]\n"
                "Observation 1: FABRICATED MODEL OBSERVATION\n"
                "Action 2: Finish[REFUTES]"
            ],
            ["Action 2: Finish[SUPPORTS]"],
        ]
    )
    agent = ActOnlyAgent(
        llm,
        GenerationConfig(),
        WikipediaEnvironment(FakeWikipediaClient(), max_steps=5),
        max_steps=5,
        prompt_builder=build_act_prompt,
        answer_normalizer=normalize_fever_label,
    )

    result = agent.predict(EXAMPLE)

    assert result.prediction == "SUPPORTS"
    assert result.tool_calls == 1
    assert result.trajectory[0].model_output == "Action 1: Search[Albert Einstein]"
    assert "FABRICATED MODEL OBSERVATION" not in llm.prompt_batches[1][0]
    assert "Opened 'Albert Einstein'" in llm.prompt_batches[1][0]
    assert llm.stop_sequences == [
        ("\nObservation 1:",),
        ("\nObservation 2:",),
    ]


def test_all_seven_fever_methods_run_a_three_claim_mock_batch() -> None:
    examples = [
        BenchmarkExample("1", "Claim one.", "SUPPORTS"),
        BenchmarkExample("2", "Claim two.", "REFUTES"),
        BenchmarkExample("3", "Claim three.", "NOT ENOUGH INFO"),
    ]
    answer_lines = [
        "Answer: SUPPORTS",
        "Answer: REFUTES",
        "Answer: NOT ENOUGH INFO",
    ]
    thought_lines = [
        "Thought: supported.\nAnswer: SUPPORTS",
        "Thought: refuted.\nAnswer: REFUTES",
        "Thought: insufficient.\nAnswer: NOT ENOUGH INFO",
    ]

    standard = StandardAgent(
        BatchLLM([answer_lines]),
        GenerationConfig(),
        prompt_builder=build_standard_prompt,
        answer_parser=parse_fever_label,
    )
    cot = CoTAgent(
        BatchLLM([thought_lines]),
        GenerationConfig(),
        prompt_builder=build_cot_prompt,
        answer_parser=parse_fever_label,
    )

    def cot_sc() -> CoTSCAgent:
        return CoTSCAgent(
            BatchLLM([thought_lines] * 21),
            GenerationConfig(temperature=0.7),
            num_samples=21,
            prompt_builder=build_cot_prompt,
            answer_parser=parse_fever_label,
            answer_normalizer=normalize_fever_label,
        )

    finish_actions = [
        "Action 1: Finish[SUPPORTS]",
        "Action 1: Finish[REFUTES]",
        "Action 1: Finish[NOT ENOUGH INFO]",
    ]
    react_actions = [
        "Thought 1: supported.\nAction 1: Finish[SUPPORTS]",
        "Thought 1: refuted.\nAction 1: Finish[REFUTES]",
        "Thought 1: insufficient.\nAction 1: Finish[NOT ENOUGH INFO]",
    ]

    def environment() -> WikipediaEnvironment:
        return WikipediaEnvironment(FakeWikipediaClient(), max_steps=5)

    act = ActOnlyAgent(
        BatchLLM([finish_actions]),
        GenerationConfig(),
        environment(),
        max_steps=5,
        environment_factory=environment,
        prompt_builder=build_act_prompt,
        answer_normalizer=normalize_fever_label,
    )
    react = ReActAgent(
        BatchLLM([react_actions]),
        GenerationConfig(),
        environment(),
        max_steps=5,
        environment_factory=environment,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )
    react_for_hybrid = ReActAgent(
        BatchLLM([react_actions]),
        GenerationConfig(),
        environment(),
        max_steps=5,
        environment_factory=environment,
        prompt_builder=build_react_prompt,
        answer_normalizer=normalize_fever_label,
    )
    react_for_fallback = StaticAgent(AgentResult("REFUTES", 1, 0, "completed"))

    agents: list[BaseAgent] = [
        standard,
        cot,
        cot_sc(),
        act,
        react,
        ReActThenCoTSCAgent(react_for_hybrid, cot_sc()),
        CoTSCThenReActAgent(cot_sc(), react_for_fallback),
    ]
    expected = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    for agent in agents:
        assert [result.prediction for result in agent.predict_batch(examples)] == expected
