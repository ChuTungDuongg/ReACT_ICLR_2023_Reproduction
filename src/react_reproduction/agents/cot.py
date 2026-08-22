"""Chain-of-Thought closed-book prompting baseline."""

from __future__ import annotations

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import parse_final_answer, parse_reasoning
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.hotpotqa import build_cot_prompt


class CoTAgent(BaseAgent):
    def __init__(self, llm: LLMProvider, generation: GenerationConfig) -> None:
        self._llm = llm
        self._generation = generation

    def predict(self, example: BenchmarkExample) -> AgentResult:
        model_output = self._llm.generate(
            build_cot_prompt(example.input_text),
            temperature=self._generation.temperature,
            top_p=self._generation.top_p,
            max_new_tokens=self._generation.max_new_tokens,
        )
        prediction = parse_final_answer(model_output)
        trajectory = (
            TrajectoryStep(
                step_index=1,
                model_output=model_output,
                thought=parse_reasoning(model_output),
            ),
        )
        return AgentResult(
            prediction=prediction,
            steps=1,
            tool_calls=0,
            termination_reason="completed" if prediction else "parsing_error",
            trajectory=trajectory,
        )
