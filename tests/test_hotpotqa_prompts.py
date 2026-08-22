"""Tests for the six paper-derived HotpotQA few-shot demonstrations."""

from react_reproduction.agents.base import TrajectoryStep
from react_reproduction.prompts.hotpotqa import (
    build_act_prompt,
    build_cot_prompt,
    build_react_prompt,
    build_standard_prompt,
)
from react_reproduction.prompts.hotpotqa_few_shot import (
    ACT_FEW_SHOT,
    COT_FEW_SHOT,
    FEW_SHOT_EXAMPLE_COUNT,
    REACT_FEW_SHOT,
    STANDARD_FEW_SHOT,
)


def test_each_method_has_exactly_six_paper_examples() -> None:
    prompt_packs = (
        STANDARD_FEW_SHOT,
        COT_FEW_SHOT,
        ACT_FEW_SHOT,
        REACT_FEW_SHOT,
    )

    assert FEW_SHOT_EXAMPLE_COUNT == 6
    assert all(pack.count("Question:") == FEW_SHOT_EXAMPLE_COUNT for pack in prompt_packs)
    assert all("Colorado orogeny" in pack for pack in prompt_packs)
    assert all("Pavel Urysohn" in pack for pack in prompt_packs)


def test_all_prompt_builders_keep_the_target_question_question_only() -> None:
    target_question = "QUESTION_ONLY_MARKER"
    prompts = (
        build_standard_prompt(target_question),
        build_cot_prompt(target_question),
        build_act_prompt(target_question, ()),
        build_react_prompt(target_question, ()),
    )

    assert all(prompt.count(target_question) == 1 for prompt in prompts)
    assert all("SECRET_SUPPORTING_CONTEXT" not in prompt for prompt in prompts)


def test_interactive_prompts_number_the_next_paper_style_step() -> None:
    trajectory = (
        TrajectoryStep(
            step_index=1,
            model_output="Action 1: Search[Albert Einstein]",
            action="Search[Albert Einstein]",
            observation="Opened 'Albert Einstein'.",
        ),
    )

    act_prompt = build_act_prompt("When was Einstein born?", trajectory)
    react_prompt = build_react_prompt("When was Einstein born?", trajectory)

    assert "Action 1: Search[Albert Einstein]" in act_prompt
    assert "Action 2: <one Search, Lookup, or Finish action>" in act_prompt
    assert "Observation 1: Opened 'Albert Einstein'." in react_prompt
    assert "Thought 2: <brief reasoning about the next action>" in react_prompt
