"""Paper-style HotpotQA prompts for all four prompting methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from react_reproduction.prompts.hotpotqa_few_shot import (
    ACT_FEW_SHOT,
    COT_FEW_SHOT,
    REACT_FEW_SHOT,
    STANDARD_FEW_SHOT,
)

if TYPE_CHECKING:
    from react_reproduction.agents.base import TrajectoryStep


def build_standard_prompt(question: str) -> str:
    return f"""Answer HotpotQA questions from your existing knowledge. Follow the six manually composed examples from Appendix C.1 of the ReAct paper.

{STANDARD_FEW_SHOT}

Question: {question}
Return exactly one line in this format:
Answer: <concise answer>"""


def build_cot_prompt(question: str) -> str:
    return f"""Answer HotpotQA questions from your existing knowledge. Follow the six manually composed Chain-of-Thought examples from Appendix C.1 of the ReAct paper.

{COT_FEW_SHOT}

Question: {question}
Return exactly two lines in this format:
Thought: <step-by-step multi-hop reasoning>
Answer: <concise answer>"""


def build_act_prompt(question: str, trajectory: Sequence[TrajectoryStep]) -> str:
    history_lines: list[str] = []
    for step in trajectory:
        if step.action:
            history_lines.append(f"Action {step.step_index}: {step.action}")
        if step.observation:
            history_lines.append(f"Observation {step.step_index}: {step.observation}")
    history = "\n".join(history_lines) or "(no previous actions)"
    next_step = len(trajectory) + 1
    return f"""Answer HotpotQA questions with Wikipedia actions and no explicit thoughts. Follow the six manually composed Act examples from Appendix C.1 of the ReAct paper.

Available actions:
- Search[entity]: open a relevant Wikipedia article.
- Lookup[text]: find the next matching sentence in the current article.
- Finish[answer]: return the concise final answer.

{ACT_FEW_SHOT}

Question: {question}

Previous action history:
{history}

Do not output Thought or Reasoning. Return exactly one line now:
Action {next_step}: <one Search, Lookup, or Finish action>"""


def build_react_prompt(question: str, trajectory: Sequence[TrajectoryStep]) -> str:
    history_lines: list[str] = []
    for step in trajectory:
        if step.thought:
            history_lines.append(f"Thought {step.step_index}: {step.thought}")
        if step.action:
            history_lines.append(f"Action {step.step_index}: {step.action}")
        if step.observation:
            history_lines.append(f"Observation {step.step_index}: {step.observation}")
    history = "\n".join(history_lines) or "(no previous steps)"
    next_step = len(trajectory) + 1
    return f"""Answer HotpotQA questions by interleaving reasoning and Wikipedia actions. Follow the six manually composed ReAct examples from Appendix C.1 of the ReAct paper.

Available actions:
- Search[entity]: open a relevant Wikipedia article.
- Lookup[text]: find the next matching sentence in the current article.
- Finish[answer]: return the concise final answer.

{REACT_FEW_SHOT}

Never invent an Observation; the environment supplies it after your action.

Question: {question}

Previous trajectory:
{history}

Return exactly two lines now:
Thought {next_step}: <brief reasoning about the next action>
Action {next_step}: <one Search, Lookup, or Finish action>"""
