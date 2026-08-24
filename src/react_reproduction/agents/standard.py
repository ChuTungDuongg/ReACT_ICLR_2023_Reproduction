"""Standard closed-book prompting baseline."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import parse_final_answer
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.hotpotqa import build_standard_prompt


class StandardAgent(BaseAgent):
    def __init__(
        self,
        llm: LLMProvider,
        generation: GenerationConfig,
        *,
        prompt_builder: Callable[[str], str] = build_standard_prompt,
        answer_parser: Callable[[str], str] = parse_final_answer,
        invalid_termination_reason: str = "parsing_error",
    ) -> None:
        self._llm = llm
        self._generation = generation
        self._prompt_builder = prompt_builder
        self._answer_parser = answer_parser
        self._invalid_termination_reason = invalid_termination_reason

    def predict(self, example: BenchmarkExample) -> AgentResult:
        return self.predict_batch((example,))[0]

    def predict_batch(
        self,
        examples: Sequence[BenchmarkExample],
    ) -> tuple[AgentResult, ...]:
        if not examples:
            return ()
        model_outputs = self._llm.generate_batch(
            [self._prompt_builder(example.input_text) for example in examples],
            temperature=self._generation.temperature,
            top_p=self._generation.top_p,
            max_new_tokens=self._generation.max_new_tokens,
        )
        if len(model_outputs) != len(examples):
            raise RuntimeError("Standard batch output count does not match inputs.")
        return tuple(self._build_result(output) for output in model_outputs)

    def _build_result(self, model_output: str) -> AgentResult:
        prediction = self._answer_parser(model_output)
        trajectory = (
            TrajectoryStep(step_index=1, model_output=model_output),
        )
        return AgentResult(
            prediction=prediction,
            steps=1,
            tool_calls=0,
            termination_reason=(
                "completed" if prediction else self._invalid_termination_reason
            ),
            trajectory=trajectory,
            metadata=(
                {}
                if prediction
                else {"parse_error": "No valid task answer could be parsed."}
            ),
        )
