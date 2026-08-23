"""Chain-of-Thought self-consistency with normalized majority voting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from react_reproduction.agents.base import AgentResult, BaseAgent, TrajectoryStep
from react_reproduction.agents.parsing import (
    parse_explicit_final_answer,
    parse_reasoning,
)
from react_reproduction.config import GenerationConfig
from react_reproduction.datasets.base import BenchmarkExample
from react_reproduction.evaluation.metrics import normalize_answer
from react_reproduction.llm.base import LLMProvider
from react_reproduction.prompts.hotpotqa import build_cot_prompt


@dataclass(frozen=True, slots=True)
class CoTSCOutcome:
    """A CoT-SC result plus the vote statistics needed by hybrid policies."""

    result: AgentResult
    sample_count: int
    valid_answer_count: int
    winning_count: int

    @property
    def confidence(self) -> float:
        return self.winning_count / self.sample_count

    @property
    def meets_paper_threshold(self) -> bool:
        """Paper fallback triggers when the winner occurs less than n/2 times."""
        return self.winning_count >= self.sample_count / 2


class CoTSCAgent(BaseAgent):
    """Sample multiple CoT answers and select the normalized plurality answer."""

    def __init__(
        self,
        llm: LLMProvider,
        generation: GenerationConfig,
        *,
        num_samples: int = 21,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if generation.temperature <= 0.0:
            raise ValueError("CoT-SC requires a positive sampling temperature.")
        self._llm = llm
        self._generation = generation
        self.num_samples = num_samples

    def predict(self, example: BenchmarkExample) -> AgentResult:
        return self.predict_with_consensus(example).result

    def predict_batch(
        self,
        examples: Sequence[BenchmarkExample],
    ) -> tuple[AgentResult, ...]:
        return tuple(
            outcome.result for outcome in self.predict_batch_with_consensus(examples)
        )

    def predict_with_consensus(self, example: BenchmarkExample) -> CoTSCOutcome:
        return self.predict_batch_with_consensus((example,))[0]

    def predict_batch_with_consensus(
        self,
        examples: Sequence[BenchmarkExample],
    ) -> tuple[CoTSCOutcome, ...]:
        if not examples:
            return ()

        trajectories: list[list[TrajectoryStep]] = [[] for _ in examples]
        answers: list[list[str]] = [[] for _ in examples]
        normalized_answers: list[list[str]] = [[] for _ in examples]

        for sample_index in range(1, self.num_samples + 1):
            model_outputs = self._llm.generate_batch(
                [build_cot_prompt(example.input_text) for example in examples],
                temperature=self._generation.temperature,
                top_p=self._generation.top_p,
                max_new_tokens=self._generation.max_new_tokens,
            )
            if len(model_outputs) != len(examples):
                raise RuntimeError(
                    "LLM batch output count does not match CoT-SC input count: "
                    f"{len(model_outputs)} != {len(examples)}."
                )
            for example_index, model_output in enumerate(model_outputs):
                answer = parse_explicit_final_answer(model_output).strip()
                normalized = normalize_answer(answer)
                if normalized:
                    answers[example_index].append(answer)
                    normalized_answers[example_index].append(normalized)
                trajectories[example_index].append(
                    TrajectoryStep(
                        step_index=sample_index,
                        model_output=model_output,
                        thought=parse_reasoning(model_output),
                        phase="cot_sc",
                    )
                )

        return tuple(
            self._build_outcome(
                example_trajectories,
                example_answers,
                example_normalized_answers,
            )
            for (
                example_trajectories,
                example_answers,
                example_normalized_answers,
            ) in zip(trajectories, answers, normalized_answers, strict=True)
        )

    def _build_outcome(
        self,
        trajectories: list[TrajectoryStep],
        answers: list[str],
        normalized_answers: list[str],
    ) -> CoTSCOutcome:
        counts = Counter(normalized_answers)
        winning_key = counts.most_common(1)[0][0] if counts else ""
        winning_count = counts[winning_key] if winning_key else 0
        prediction = next(
            (
                answer
                for answer, normalized in zip(
                    answers, normalized_answers, strict=True
                )
                if normalized == winning_key
            ),
            "",
        )
        valid_answer_count = len(answers)
        metadata = {
            "cot_sc_samples": self.num_samples,
            "cot_sc_valid_answers": valid_answer_count,
            "cot_sc_winning_count": winning_count,
            "cot_sc_confidence": winning_count / self.num_samples,
            "cot_sc_winning_answer": prediction,
            "cot_sc_vote_counts": dict(counts.most_common()),
        }
        result = AgentResult(
            prediction=prediction,
            steps=len(trajectories),
            tool_calls=0,
            termination_reason=(
                "cot_sc_consensus" if prediction else "cot_sc_no_answer"
            ),
            trajectory=tuple(trajectories),
            metadata=metadata,
        )
        return CoTSCOutcome(
            result=result,
            sample_count=self.num_samples,
            valid_answer_count=valid_answer_count,
            winning_count=winning_count,
        )
