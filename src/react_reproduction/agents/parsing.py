"""Parsing helpers for structured prompting outputs."""

from __future__ import annotations

import re

from react_reproduction.tools.wikipedia import ActionType, ToolAction


_FINAL_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*(.+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_REASONING = re.compile(
    r"(?:^|\n)\s*reasoning\s*:\s*(.*?)(?=\n\s*final\s+answer\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_TOOL_ACTION = re.compile(
    r"(?:^|\n)\s*(?:action\s*:\s*)?"
    r"(search|lookup|finish)\s*\[(.*?)\]\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_THOUGHT = re.compile(
    r"(?:^|\n)\s*thought\s*:\s*(.*?)(?=\n\s*(?:action\s*:)?"
    r"\s*(?:search|lookup|finish)\s*\[|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


class ActionParseError(ValueError):
    """Raised when a model output contains no valid environment action."""


def parse_final_answer(model_output: str) -> str:
    """Extract a concise answer while tolerating minor format drift."""
    matches = _FINAL_ANSWER.findall(model_output.strip())
    if matches:
        return matches[-1].strip()

    non_empty_lines = [line.strip() for line in model_output.splitlines() if line.strip()]
    if not non_empty_lines:
        return ""
    return non_empty_lines[-1].strip()


def parse_reasoning(model_output: str) -> str | None:
    match = _REASONING.search(model_output.strip())
    if match is None:
        return None
    reasoning = match.group(1).strip()
    return reasoning or None


def parse_tool_action(model_output: str) -> ToolAction:
    """Parse the first Search/Lookup/Finish action in a model completion."""
    matches = _TOOL_ACTION.findall(model_output.strip())
    if not matches:
        raise ActionParseError(
            "Expected one action in the form Search[...], Lookup[...], or Finish[...]."
        )
    action_name, argument = matches[0]
    normalized_name = action_name.casefold()
    cleaned_argument = argument.strip()
    named_argument = {
        "search": "entity",
        "lookup": "text",
        "finish": "answer",
    }[normalized_name]
    named_match = re.match(
        rf"^{named_argument}\s*(?:=|:)\s*(.+)$",
        cleaned_argument,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if named_match is not None:
        cleaned_argument = named_match.group(1).strip()
    if (
        len(cleaned_argument) >= 2
        and cleaned_argument[0] == cleaned_argument[-1]
        and cleaned_argument[0] in {"'", '"'}
    ):
        cleaned_argument = cleaned_argument[1:-1].strip()
    if not cleaned_argument:
        raise ActionParseError("Tool action argument cannot be empty.")
    action_type = {
        "search": ActionType.SEARCH,
        "lookup": ActionType.LOOKUP,
        "finish": ActionType.FINISH,
    }[normalized_name]
    return ToolAction(action_type, cleaned_argument)


def parse_thought(model_output: str) -> str | None:
    """Extract the first explicit ReAct thought, if the model emitted one."""
    matches = _THOUGHT.findall(model_output.strip())
    if not matches:
        return None
    thought = matches[0].strip()
    return thought or None


def parse_react_output(model_output: str) -> tuple[str | None, ToolAction]:
    """Parse one ReAct turn while tolerating an omitted Thought label."""
    return parse_thought(model_output), parse_tool_action(model_output)
