"""Language-model provider interface used by all prompting agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class LLMProvider(ABC):
    """Minimal model interface that keeps agents backend-independent."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        """Generate one completion for a fully constructed prompt."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        """Generate a batch, with a sequential fallback for simple providers."""
        return tuple(
            self.generate(
                prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for prompt in prompts
        )
