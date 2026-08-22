"""HotpotQA prompts for closed-book baselines."""

from __future__ import annotations

from typing import Sequence

from react_reproduction.agents.base import TrajectoryStep


def build_standard_prompt(question: str) -> str:
    return f"""Answer the following HotpotQA question from your existing knowledge.
Return only one line in this exact format:
Final Answer: <concise answer>

Question: {question}
"""


def build_cot_prompt(question: str) -> str:
    return f"""Answer the following HotpotQA question from your existing knowledge.
Reason step by step, then give a concise final answer.
Use exactly this format:
Reasoning: <brief multi-hop reasoning>
Final Answer: <concise answer>

Question: {question}
"""


def build_act_prompt(question: str, trajectory: Sequence[TrajectoryStep]) -> str:
    history_lines: list[str] = []
    for step in trajectory:
        if step.action:
            history_lines.append(f"Action: {step.action}")
        if step.observation:
            history_lines.append(f"Observation: {step.observation}")
    history = "\n".join(history_lines) or "(no previous actions)"
    return f"""Answer the HotpotQA question using Wikipedia actions.
Available actions:
- Search[entity]: open a relevant Wikipedia article.
- Lookup[text]: find the next matching sentence in the current article.
- Finish[answer]: return the concise final answer.

Do not output Thought or Reasoning. Return exactly one Action line.

Question: {question}

Previous action history:
{history}

Action:"""


def build_react_prompt(question: str, trajectory: Sequence[TrajectoryStep]) -> str:
    history_lines: list[str] = []
    for step in trajectory:
        if step.thought:
            history_lines.append(f"Thought: {step.thought}")
        if step.action:
            history_lines.append(f"Action: {step.action}")
        if step.observation:
            history_lines.append(f"Observation: {step.observation}")
    history = "\n".join(history_lines) or "(no previous steps)"
    return f"""Answer the HotpotQA question by interleaving reasoning and Wikipedia actions.
Available actions:
- Search[entity]: open a relevant Wikipedia article.
- Lookup[text]: find the next matching sentence in the current article.
- Finish[answer]: return the concise final answer.

At every step return exactly two lines:
Thought: <brief reasoning about the next action>
Action: <one Search, Lookup, or Finish action>

Never invent an Observation; the environment supplies it after your action.

Question: {question}

Previous trajectory:
{history}

Return the next Thought and Action now."""
