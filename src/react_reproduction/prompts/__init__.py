"""Task- and method-specific prompt templates."""

from react_reproduction.prompts.hotpotqa import (
    build_act_prompt,
    build_cot_prompt,
    build_react_prompt,
    build_standard_prompt,
)

__all__ = [
    "build_act_prompt",
    "build_cot_prompt",
    "build_react_prompt",
    "build_standard_prompt",
]
