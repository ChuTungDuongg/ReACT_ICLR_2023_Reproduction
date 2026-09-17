"""Language-model provider interface used by all prompting agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from inspect import Parameter, signature
from typing import Any


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
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Generate one completion for a fully constructed prompt."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop_sequences: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Generate a batch, with a sequential fallback for simple providers."""
        outputs: list[str] = []
        supports_stops = _accepts_stop_sequences(self.generate)
        for prompt in prompts:
            kwargs: dict[str, Any] = {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            }
            if stop_sequences is not None and supports_stops:
                kwargs["stop_sequences"] = stop_sequences
            output = self.generate(prompt, **kwargs)
            outputs.append(truncate_at_stop_sequences(output, stop_sequences))
        return tuple(outputs)


def normalize_stop_sequences(
    stop_sequences: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate and freeze optional stop strings for consistent provider behavior."""
    if stop_sequences is None:
        return ()
    if isinstance(stop_sequences, str):
        stop_sequences = (stop_sequences,)
    normalized = tuple(stop_sequences)
    if any(not isinstance(stop, str) for stop in normalized):
        raise TypeError("stop_sequences must contain only strings.")
    if any(not stop for stop in normalized):
        raise ValueError("stop_sequences cannot contain an empty string.")
    return normalized


def truncate_at_stop_sequences(
    completion: str,
    stop_sequences: Sequence[str] | None,
) -> str:
    """Cut a completion at the earliest requested stop string."""
    stops = normalize_stop_sequences(stop_sequences)
    indexes = [completion.find(stop) for stop in stops if stop in completion]
    if indexes:
        completion = completion[: min(indexes)]
    return completion


def generate_batch_with_stops(
    provider: LLMProvider,
    prompts: Sequence[str],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: Sequence[str] | None,
) -> tuple[str, ...]:
    """Use stop-aware providers while retaining compatibility with older subclasses."""
    stops = normalize_stop_sequences(stop_sequences)
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    if stops and _accepts_stop_sequences(provider.generate_batch):
        kwargs["stop_sequences"] = stops
    outputs = provider.generate_batch(prompts, **kwargs)
    return tuple(truncate_at_stop_sequences(output, stops) for output in outputs)


def _accepts_stop_sequences(callable_object: Any) -> bool:
    parameters = signature(callable_object).parameters
    return "stop_sequences" in parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
