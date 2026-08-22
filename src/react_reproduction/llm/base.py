"""Language-model provider interface used by all prompting agents."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
