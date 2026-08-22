"""Tests for Search, Lookup, and Finish action parsing."""

import pytest

from react_reproduction.agents.parsing import (
    ActionParseError,
    parse_react_output,
    parse_tool_action,
)
from react_reproduction.tools.wikipedia import ActionType


@pytest.mark.parametrize(
    ("text", "expected_type", "expected_argument"),
    [
        ("Action: Search[Albert Einstein]", ActionType.SEARCH, "Albert Einstein"),
        (
            "Action: Search[entity=University of Saint Petersburg]",
            ActionType.SEARCH,
            "University of Saint Petersburg",
        ),
        (
            "Action: Search[entity:Sergei Aleksandrovich Tokarev]",
            ActionType.SEARCH,
            "Sergei Aleksandrovich Tokarev",
        ),
        ("Lookup[born]", ActionType.LOOKUP, "born"),
        (
            'Action: Lookup[text="Sergei Aleksandrovich Tokarev"]',
            ActionType.LOOKUP,
            "Sergei Aleksandrovich Tokarev",
        ),
        ("Action: Finish[Germany]", ActionType.FINISH, "Germany"),
        ('Finish[answer="1879"]', ActionType.FINISH, "1879"),
    ],
)
def test_parse_tool_action(
    text: str,
    expected_type: ActionType,
    expected_argument: str,
) -> None:
    action = parse_tool_action(text)
    assert action.action_type is expected_type
    assert action.argument == expected_argument


def test_parse_tool_action_rejects_unstructured_text() -> None:
    with pytest.raises(ActionParseError, match="Expected one action"):
        parse_tool_action("I should search Wikipedia now.")


def test_parse_react_output_extracts_thought_and_action() -> None:
    thought, action = parse_react_output(
        "Thought: I should find Einstein's page.\nAction: Search[Albert Einstein]"
    )
    assert thought == "I should find Einstein's page."
    assert action.action_type is ActionType.SEARCH


def test_parse_react_output_tolerates_action_only() -> None:
    thought, action = parse_react_output("Action: Finish[1879]")
    assert thought is None
    assert action.argument == "1879"


def test_parser_executes_only_first_action_when_model_emits_multiple() -> None:
    action = parse_tool_action(
        "Action: Search[Albert Einstein]\nAction: Lookup[born]"
    )
    assert action.action_type is ActionType.SEARCH
    assert action.argument == "Albert Einstein"
