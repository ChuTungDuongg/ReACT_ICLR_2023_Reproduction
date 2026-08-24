"""Paper-fidelity tests for the three Appendix C.2 FEVER exemplars."""

from react_reproduction.agents.base import TrajectoryStep
from react_reproduction.prompts.fever import (
    FEVER_EXEMPLARS,
    FEVER_PROMPT_VERSION,
    build_act_prompt,
    build_cot_prompt,
    build_react_prompt,
    build_standard_prompt,
)


PAPER_CLAIMS = (
    "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
    "Stranger Things is set in Bloomington, Indiana.",
    "Beautiful reached number two on the Billboard Hot 100 in 2003.",
)


def test_exact_three_appendix_c2_exemplars_cover_all_labels() -> None:
    assert tuple(example.claim for example in FEVER_EXEMPLARS) == PAPER_CLAIMS
    assert tuple(example.label for example in FEVER_EXEMPLARS) == (
        "SUPPORTS",
        "REFUTES",
        "NOT ENOUGH INFO",
    )
    assert FEVER_PROMPT_VERSION == "fever-appendix-c2-v1"


def test_all_four_prompts_use_the_same_three_claims() -> None:
    prompts = (
        build_standard_prompt("TARGET"),
        build_cot_prompt("TARGET"),
        build_act_prompt("TARGET", ()),
        build_react_prompt("TARGET", ()),
    )
    for prompt in prompts:
        assert all(prompt.count(claim) == 1 for claim in PAPER_CLAIMS)
        assert prompt.count("TARGET") == 1


def test_standard_is_claim_answer_ablation() -> None:
    prompt = build_standard_prompt("TARGET")
    assert "Thought" not in prompt
    assert "Action" not in prompt
    assert "Observation 1:" not in prompt
    assert prompt.count("Claim:") == 4
    assert prompt.endswith("Claim: TARGET\nAnswer:")


def test_cot_is_reasoning_only_ablation() -> None:
    prompt = build_cot_prompt("TARGET")
    assert "Thought:" in prompt
    assert "Action" not in prompt
    assert "Observation 1:" not in prompt
    assert "Search[" not in prompt
    assert prompt.endswith("Claim: TARGET\nThought:")


def test_act_removes_thoughts_but_keeps_actions_and_observations() -> None:
    prompt = build_act_prompt("TARGET", ())
    assert "Thought" not in prompt
    assert "Search[Nikolaj Coster-Waldau]" in prompt
    assert "Lookup[Billboard Hot 100]" in prompt
    assert "Observation 1:" in prompt
    assert prompt.endswith("Action 1:")


def test_react_keeps_dense_thought_action_observation_structure() -> None:
    history = (
        TrajectoryStep(
            step_index=1,
            model_output="Thought 1: verify\nAction 1: Search[X]",
            thought="verify",
            action="Search[X]",
            observation="Opened X.",
        ),
    )
    prompt = build_react_prompt("TARGET", history)
    assert "Thought 1: I need to search Nikolaj" in prompt
    assert "Action 1: Search[Nikolaj Coster-Waldau]" in prompt
    assert "Observation 1: Nikolaj William" in prompt
    assert "Thought 1: verify" in prompt
    assert prompt.endswith("Thought 2:")


def test_target_prompt_never_receives_gold_evidence() -> None:
    marker = "SECRET_GOLD_EVIDENCE"
    assert marker not in build_standard_prompt("TARGET")
    assert marker not in build_cot_prompt("TARGET")
    assert marker not in build_act_prompt("TARGET", ())
    assert marker not in build_react_prompt("TARGET", ())
