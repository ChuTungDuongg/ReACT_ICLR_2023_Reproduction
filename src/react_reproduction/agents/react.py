"""ReAct agent interleaving explicit reasoning with Wikipedia actions."""

from __future__ import annotations

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import (
    ActionParseError,
    parse_react_output,
    parse_thought,
)
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.hotpotqa import build_react_prompt
from react_reproduction.tools.wikipedia import WikipediaEnvironment


class ReActAgent(BaseAgent):
    """Run the Thought -> Action -> Observation loop for one question."""

    def __init__(
        self,
        llm: LLMProvider,
        generation: GenerationConfig,
        environment: WikipediaEnvironment,
        *,
        max_steps: int,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._llm = llm
        self._generation = generation
        self._environment = environment
        self._max_steps = max_steps

    def predict(self, example: BenchmarkExample) -> AgentResult:
        self._environment.reset()
        trajectory: list[TrajectoryStep] = []
        tool_calls = 0
        last_parse_failed = False

        for step_index in range(1, self._max_steps + 1):
            model_output = self._llm.generate(
                build_react_prompt(example.input_text, trajectory),
                temperature=self._generation.temperature,
                top_p=self._generation.top_p,
                max_new_tokens=self._generation.max_new_tokens,
            )
            try:
                thought, action = parse_react_output(model_output)
            except ActionParseError as error:
                last_parse_failed = True
                trajectory.append(
                    TrajectoryStep(
                        step_index=step_index,
                        model_output=model_output,
                        thought=parse_thought(model_output),
                        observation=f"Invalid action format: {error}",
                    )
                )
                continue

            last_parse_failed = False
            execution = self._environment.execute(action)
            tool_calls += int(execution.tool_called)
            trajectory.append(
                TrajectoryStep(
                    step_index=step_index,
                    model_output=model_output,
                    thought=thought,
                    action=action.canonical,
                    observation=execution.observation,
                )
            )
            if execution.terminated:
                return AgentResult(
                    prediction=execution.answer or "",
                    steps=len(trajectory),
                    tool_calls=tool_calls,
                    termination_reason=execution.termination_reason or "terminated",
                    trajectory=tuple(trajectory),
                )

        return AgentResult(
            prediction="",
            steps=len(trajectory),
            tool_calls=tool_calls,
            termination_reason=(
                "parsing_error" if last_parse_failed else "max_steps_exceeded"
            ),
            trajectory=tuple(trajectory),
        )
