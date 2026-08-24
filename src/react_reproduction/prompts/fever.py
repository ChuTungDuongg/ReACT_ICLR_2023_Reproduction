"""Paper-faithful FEVER prompts from Appendix C.2 and the official source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from react_reproduction.agents.base import TrajectoryStep


FEVER_PROMPT_VERSION = "fever-appendix-c2-v1"
FEVER_INSTRUCTION = (
    "Determine if there is Observation that SUPPORTS or REFUTES a Claim, "
    "or if there is NOT ENOUGH INFORMATION."
)


@dataclass(frozen=True, slots=True)
class FeverDemoStep:
    thought: str
    action: str
    observation: str | None = None


@dataclass(frozen=True, slots=True)
class FeverExemplar:
    claim: str
    label: str
    cot_thought: str
    steps: tuple[FeverDemoStep, ...]


FEVER_EXEMPLARS = (
    FeverExemplar(
        claim="Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        label="SUPPORTS",
        cot_thought=(
            "Nikolaj William Coster-Waldau appeared in the 2009 Fox television "
            "film Virtuality, so he has worked with the Fox Broadcasting Company."
        ),
        steps=(
            FeverDemoStep(
                thought=(
                    "I need to search Nikolaj Coster-Waldau and find if he has "
                    "worked with the Fox Broadcasting Company."
                ),
                action="Search[Nikolaj Coster-Waldau]",
                observation=(
                    "Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish "
                    "actor and producer. He graduated from the Danish National "
                    "School of Performing Arts in Copenhagen in 1993,[1] and had his "
                    "breakthrough role in Denmark with the film Nightwatch (1994). "
                    "He played Jaime Lannister in the HBO fantasy drama series Game "
                    "of Thrones, for which he received two Primetime Emmy Award "
                    "nominations for Outstanding Supporting Actor in a Drama Series.. "
                    "Coster-Waldau has appeared in numerous films in his native "
                    "Denmark and Scandinavia, including Headhunters (2011) and A "
                    "Thousand Times Good Night (2013). In the U.S, his debut film "
                    "role was in the war film Black Hawk Down (2001), playing Medal "
                    "of Honor recipient Gary Gordon.[2] He then played a detective "
                    "in the short-lived Fox television series New Amsterdam (2008), "
                    "and appeared in the 2009 Fox television film Virtuality, "
                    "originally intended as a pilot."
                ),
            ),
            FeverDemoStep(
                thought=(
                    'Because he "appeared in the 2009 Fox television film '
                    'Virtuality", he should have worked with the Fox Broadcasting '
                    "Company."
                ),
                action="Finish[SUPPORTS]",
            ),
        ),
    ),
    FeverExemplar(
        claim="Stranger Things is set in Bloomington, Indiana.",
        label="REFUTES",
        cot_thought=(
            "Stranger Things is in the fictional town of Hawkins, Indiana, not in "
            "Bloomington, Indiana."
        ),
        steps=(
            FeverDemoStep(
                thought=(
                    "I should search for Stranger Things, and see if it is set in "
                    "Bloomington, Indiana."
                ),
                action="Search[Stranger Things]",
                observation=(
                    "Stranger Things is an American science fiction horror drama "
                    "television series created by the Duffer Brothers. Set in the "
                    "1980s, primarily in the fictional town of Hawkins, Indiana, the "
                    "series centers on a number of mysteries and supernatural events "
                    "occurring around the town and their impact on an ensemble of "
                    "child and adult characters."
                ),
            ),
            FeverDemoStep(
                thought=(
                    'The observation says that it is set in a "fictional town of '
                    'Hawkins, Indiana", so it is not set in Bloomington.'
                ),
                action="Finish[REFUTES]",
            ),
        ),
    ),
    FeverExemplar(
        claim="Beautiful reached number two on the Billboard Hot 100 in 2003.",
        label="NOT ENOUGH INFO",
        cot_thought=(
            "The song peaked at number two on the Billboard Hot 100 in the United "
            "States, but not sure if it was in 2003."
        ),
        steps=(
            FeverDemoStep(
                thought=(
                    "I need to search the song Beautiful and find if it reached "
                    "number two on the Billboard Hot 100 in 2003."
                ),
                action="Search[Beautiful]",
                observation=(
                    "Could not find [Beautiful]. Similar: ['Beautiful', 'Beautiful, "
                    "Beautiful', 'A Beautiful Mind (film)', 'Beautiful (Christina "
                    "Aguilera song)', 'Life Is Beautiful']."
                ),
            ),
            FeverDemoStep(
                thought=(
                    'From suggestions, I should search "Beautiful (Christina '
                    'Aguilera song)" to find the song.'
                ),
                action="Search[Beautiful (Christina Aguilera song)]",
                observation=(
                    '"Beautiful" is a song recorded by American singer Christina '
                    "Aguilera for her fourth studio album, Stripped (2002)."
                ),
            ),
            FeverDemoStep(
                thought=(
                    'It does not mention Billboard, so I need to look up "Billboard '
                    'Hot 100" to find if it reached number two on it in 2003.'
                ),
                action="Lookup[Billboard Hot 100]",
                observation=(
                    "(Result 1 / 3) The song peaked at number two on the Billboard "
                    "Hot 100 in the United States, where it was certified Gold for "
                    "500,000 units shipped."
                ),
            ),
            FeverDemoStep(
                thought=(
                    "It only says the song peaked at number two on the Billboard Hot "
                    "100, but not if it was in 2003. I am not sure if this claim is "
                    "true or not."
                ),
                action="Finish[NOT ENOUGH INFO]",
            ),
        ),
    ),
)


def build_standard_prompt(claim: str) -> str:
    return _join_prompt(_render_standard_examples(), f"Claim: {claim}\nAnswer:")


def build_cot_prompt(claim: str) -> str:
    return _join_prompt(_render_cot_examples(), f"Claim: {claim}\nThought:")


def build_act_prompt(
    claim: str,
    trajectory: Sequence[TrajectoryStep],
    *,
    force_finish: bool = False,
) -> str:
    history = _render_history(trajectory, include_thought=False)
    next_step = len(trajectory) + 1
    suffix = f"Claim: {claim}\n{history}"
    if force_finish:
        suffix += (
            "\nThis is the final allowed step. Return one of SUPPORTS, REFUTES, "
            "or NOT ENOUGH INFO."
        )
    suffix += f"\nAction {next_step}:"
    return _join_prompt(_render_act_examples(), suffix)


def build_react_prompt(
    claim: str,
    trajectory: Sequence[TrajectoryStep],
    *,
    force_finish: bool = False,
) -> str:
    history = _render_history(trajectory, include_thought=True)
    next_step = len(trajectory) + 1
    suffix = f"Claim: {claim}\n{history}"
    if force_finish:
        suffix += (
            "\nThis is the final allowed step. Use Finish with SUPPORTS, REFUTES, "
            "or NOT ENOUGH INFO."
        )
    suffix += f"\nThought {next_step}:"
    return _join_prompt(_render_react_examples(), suffix)


def _render_standard_examples() -> str:
    return "\n\n".join(
        f"Claim: {example.claim}\nAnswer: {example.label}"
        for example in FEVER_EXEMPLARS
    )


def _render_cot_examples() -> str:
    return "\n\n".join(
        f"Claim: {example.claim}\nThought: {example.cot_thought}\n"
        f"Answer: {example.label}"
        for example in FEVER_EXEMPLARS
    )


def _render_act_examples() -> str:
    rendered: list[str] = []
    for example in FEVER_EXEMPLARS:
        lines = [f"Claim: {example.claim}"]
        for index, step in enumerate(example.steps, start=1):
            lines.append(f"Action {index}: {step.action}")
            if step.observation is not None:
                lines.append(f"Observation {index}: {step.observation}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _render_react_examples() -> str:
    rendered: list[str] = []
    for example in FEVER_EXEMPLARS:
        lines = [f"Claim: {example.claim}"]
        for index, step in enumerate(example.steps, start=1):
            lines.append(f"Thought {index}: {step.thought}")
            lines.append(f"Action {index}: {step.action}")
            if step.observation is not None:
                lines.append(f"Observation {index}: {step.observation}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _render_history(
    trajectory: Sequence[TrajectoryStep],
    *,
    include_thought: bool,
) -> str:
    lines: list[str] = []
    for step in trajectory:
        if include_thought and step.thought:
            lines.append(f"Thought {step.step_index}: {step.thought}")
        if step.action:
            lines.append(f"Action {step.step_index}: {step.action}")
        if step.observation:
            lines.append(f"Observation {step.step_index}: {step.observation}")
    return "\n".join(lines)


def _join_prompt(examples: str, target: str) -> str:
    return f"{FEVER_INSTRUCTION}\n\n{examples}\n\n{target}"
