"""Task prompt suites used to keep all agents task-neutral."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from react_reproduction.agents.base import TrajectoryStep
from react_reproduction.prompts import fever, hotpotqa


InteractivePromptBuilder = Callable[
    [str, Sequence[TrajectoryStep]],
    str,
]


@dataclass(frozen=True, slots=True)
class PromptSuite:
    version: str
    standard: Callable[[str], str]
    cot: Callable[[str], str]
    act: Callable[..., str]
    react: Callable[..., str]


PROMPT_SUITES = {
    "hotpotqa": PromptSuite(
        version="hotpotqa-appendix-c1-v1",
        standard=hotpotqa.build_standard_prompt,
        cot=hotpotqa.build_cot_prompt,
        act=hotpotqa.build_act_prompt,
        react=hotpotqa.build_react_prompt,
    ),
    "fever": PromptSuite(
        version=fever.FEVER_PROMPT_VERSION,
        standard=fever.build_standard_prompt,
        cot=fever.build_cot_prompt,
        act=fever.build_act_prompt,
        react=fever.build_react_prompt,
    ),
}


def get_prompt_suite(task: str) -> PromptSuite:
    try:
        return PROMPT_SUITES[task]
    except KeyError as error:
        raise ValueError(f"No prompt suite registered for task {task!r}.") from error
