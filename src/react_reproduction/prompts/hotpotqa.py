"""Paper-style HotpotQA prompts for the four base prompting methods."""

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


def build_act_prompt(
    question: str,
    trajectory: Sequence[TrajectoryStep],
    *,
    force_finish: bool = False,
) -> str:
    history_lines: list[str] = []
    for step in trajectory:
        if step.action:
            history_lines.append(f"Action {step.step_index}: {step.action}")
        if step.observation:
            history_lines.append(f"Observation {step.step_index}: {step.observation}")
    history = "\n".join(history_lines) or "(no previous actions)"
    next_step = len(trajectory) + 1
    next_action_instruction = (
        "This is the final allowed step. Do not call Search or Lookup. "
        "Use the available evidence and return exactly one concise "
        f"Finish action now:\nAction {next_step}: Finish[<best answer>]"
        if force_finish
        else (
            "Return exactly one line now:\n"
            f"Action {next_step}: <one Search, Lookup, or Finish action>"
        )
    )
    return f"""Answer HotpotQA questions with Wikipedia actions and no explicit thoughts. Follow the six manually composed Act examples from Appendix C.1 of the ReAct paper.

Available actions:
- Search[entity]: open a Wikipedia article by concise entity or page title.
- Lookup[text]: find the next sentence containing literal text in the current article.
- Finish[answer]: return the concise final answer.

Tool strategy:
- Search is not a general web search. Never put a full question or an attribute
  query such as "entity founding year" inside Search.
- Once the relevant article is open, use Lookup with a short literal keyword
  such as "founded", "members", "born", or "also known" to find a detail.
- If Search lists the desired article as a candidate, Search its exact title.
- Never repeat an identical Search. Simplify to a canonical entity, switch to
  Lookup, choose a different candidate, or Finish.
- Match the requested answer type. Do not answer yes/no unless the question is
  yes/no, and Finish as soon as the observations support a concise answer.

{ACT_FEW_SHOT}

Question: {question}

Previous action history:
{history}

Do not output Thought or Reasoning. {next_action_instruction}"""


def build_react_prompt(
    question: str,
    trajectory: Sequence[TrajectoryStep],
    *,
    force_finish: bool = False,
) -> str:
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
    next_action_instruction = (
        "This is the final allowed step. Do not call Search or Lookup. Use the "
        "available evidence, make the best supported choice, and return "
        f"Finish now:\nThought {next_step}: <brief final reasoning>\n"
        f"Action {next_step}: Finish[<best concise answer>]"
        if force_finish
        else (
            "Return exactly two lines now:\n"
            f"Thought {next_step}: <brief reasoning about the next action>\n"
            f"Action {next_step}: <one Search, Lookup, or Finish action>"
        )
    )
    return f"""Answer HotpotQA questions by interleaving reasoning and Wikipedia actions. Follow the six manually composed ReAct examples from Appendix C.1 of the ReAct paper.

Available actions:
- Search[entity]: open a Wikipedia article by concise entity or page title.
- Lookup[text]: find the next sentence containing literal text in the current article.
- Finish[answer]: return the concise final answer.

Tool strategy:
- Search is not a general web search. Never put a full question or an attribute
  query such as "entity founding year" inside Search.
- Once the relevant article is open, use Lookup with a short literal keyword
  such as "founded", "members", "born", or "also known" to find a detail.
- If Search lists the desired article as a candidate, Search its exact title.
- Never repeat an identical Search. Simplify to a canonical entity, switch to
  Lookup, choose a different candidate, or Finish.
- Match the requested answer type. Do not answer yes/no unless the question is
  yes/no, and Finish as soon as the observations support a concise answer.

{REACT_FEW_SHOT}

Never invent an Observation; the environment supplies it after your action.

Question: {question}

Previous trajectory:
{history}

{next_action_instruction}"""
