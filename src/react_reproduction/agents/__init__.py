"""Prompting agents implemented directly without an agent framework."""

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.act import ActOnlyAgent
from react_reproduction.agents.cot import CoTAgent
from react_reproduction.agents.cot_sc import CoTSCAgent, CoTSCOutcome
from react_reproduction.agents.hybrid import CoTSCThenReActAgent, ReActThenCoTSCAgent
from react_reproduction.agents.react import ReActAgent
from react_reproduction.agents.standard import StandardAgent

__all__ = [
    "AgentResult",
    "ActOnlyAgent",
    "BaseAgent",
    "CoTAgent",
    "CoTSCAgent",
    "CoTSCOutcome",
    "CoTSCThenReActAgent",
    "ReActAgent",
    "ReActThenCoTSCAgent",
    "StandardAgent",
    "TrajectoryStep",
]
