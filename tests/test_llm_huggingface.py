"""Unit tests for Hugging Face generation semantics without loading a model."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from react_reproduction.llm.huggingface import HuggingFaceProvider


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "prefix STOP later"

    def batch_decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert skip_special_tokens is True
        return ["prefix STOP later", "uninterrupted"]


class FakeModel:
    def __init__(self, output_ids: torch.Tensor) -> None:
        self.output_ids = output_ids
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        return self.output_ids


class FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class ChatTokenizer:
    chat_template = "test-template"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def apply_chat_template(self, conversations: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((conversations, kwargs))
        batch_size = len(conversations) if conversations and isinstance(
            conversations[0], list
        ) else 1
        return {"input_ids": torch.ones((batch_size, 2), dtype=torch.long)}


def _provider(output_ids: torch.Tensor) -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._torch = FakeTorch()
    provider.tokenizer = FakeTokenizer()
    provider.model = FakeModel(output_ids)
    provider._prepare_inputs = lambda prompt: {"input_ids": torch.tensor([[10, 11]])}
    provider._prepare_batch_inputs = lambda prompts: {
        "input_ids": torch.tensor([[10, 11], [20, 21]])
    }
    return provider


def test_single_and_batch_generation_share_limits_sampling_stops_and_decoding() -> None:
    single = _provider(torch.tensor([[10, 11, 30, 31, 32]]))
    batch = _provider(
        torch.tensor(
            [
                [10, 11, 30, 31, 32],
                [20, 21, 40, 41, 42],
            ]
        )
    )

    single_output = single.generate(
        "prompt",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=17,
        stop_sequences=("STOP",),
    )
    batch_outputs = batch.generate_batch(
        ["prompt", "other prompt"],
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=17,
        stop_sequences=("STOP",),
    )

    assert single_output == "prefix"
    assert batch_outputs == ("prefix", "uninterrupted")
    for call in (single.model.calls[0], batch.model.calls[0]):
        assert call["max_new_tokens"] == 17
        assert call["do_sample"] is True
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.9
        assert call["pad_token_id"] == 0
        assert "stopping_criteria" in call


def test_huggingface_stop_sequences_are_optional_and_earliest_stop_wins() -> None:
    provider = _provider(torch.tensor([[10, 11, 30, 31, 32]]))
    provider.tokenizer.decode = lambda *args, **kwargs: "before SECOND then FIRST"

    output = provider.generate(
        "prompt",
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=8,
        stop_sequences=("FIRST", "SECOND"),
    )

    assert output == "before"
    assert provider.model.calls[0]["max_new_tokens"] == 8
    assert provider.model.calls[0]["do_sample"] is False
    assert "temperature" not in provider.model.calls[0]
    assert "top_p" not in provider.model.calls[0]


def test_single_and_batch_chat_templates_use_the_same_generation_contract() -> None:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._torch = torch
    provider.device = "cpu"
    provider.model = object()
    provider.tokenizer = ChatTokenizer()

    single = provider._prepare_inputs("first prompt")
    batch = provider._prepare_batch_inputs(("first prompt", "second prompt"))

    assert single["input_ids"].shape == (1, 2)
    assert batch["input_ids"].shape == (2, 2)
    single_conversation, single_kwargs = provider.tokenizer.calls[0]
    batch_conversations, batch_kwargs = provider.tokenizer.calls[1]
    assert single_conversation == [{"role": "user", "content": "first prompt"}]
    assert batch_conversations == [
        [{"role": "user", "content": "first prompt"}],
        [{"role": "user", "content": "second prompt"}],
    ]
    assert single_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert batch_kwargs == {
        **single_kwargs,
        "padding": True,
    }


def test_batch_stopping_criteria_tracks_each_completion_independently() -> None:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider._torch = torch
    provider.tokenizer = FakeTokenizer()
    criteria = provider._stopping_criteria(("STOP",), input_length=2)
    assert criteria is not None
    stop_tokens = provider.tokenizer.encode("STOP", add_special_tokens=False)
    unfinished_tokens = provider.tokenizer.encode("STAY", add_special_tokens=False)
    input_ids = torch.tensor(
        [
            [10, 11, *stop_tokens],
            [20, 21, *unfinished_tokens],
        ]
    )

    stopped = criteria(input_ids, scores=None)

    assert stopped.tolist() == [True, False]
