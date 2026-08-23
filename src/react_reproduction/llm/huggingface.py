"""Hugging Face Transformers implementation of :class:`LLMProvider`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from react_reproduction.llm.base import LLMProvider


LOGGER = logging.getLogger("react_reproduction.llm.huggingface")
SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}


class HuggingFaceProvider(LLMProvider):
    """Run instruction-tuned causal language models locally or in Colab."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        cache_dir: Path | None = None,
        seed: int = 42,
        trust_remote_code: bool = False,
    ) -> None:
        if device not in SUPPORTED_DEVICES:
            supported = ", ".join(sorted(SUPPORTED_DEVICES))
            raise ValueError(f"Unsupported device {device!r}; expected one of {supported}.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Hugging Face inference requires torch and transformers. "
                "Install dependencies with 'pip install -r requirements.txt'."
            ) from error

        self._torch = torch
        self.model_name = model_name
        self.device = _resolve_device(device, torch)
        self._set_seed(seed)
        resolved_cache = str(cache_dir.resolve()) if cache_dir is not None else None

        LOGGER.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=resolved_cache,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = _preferred_dtype(self.device, torch)
        model_kwargs: dict[str, Any] = {
            "cache_dir": resolved_cache,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        LOGGER.info(
            "Loading model: %s device=%s dtype=%s",
            model_name,
            self.device,
            dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        LOGGER.info("Model ready: %s", model_name)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        model_inputs = self._prepare_inputs(prompt)
        input_length = model_inputs["input_ids"].shape[-1]
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0.0:
            generate_kwargs.update(temperature=temperature, top_p=top_p)

        with self._torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        completion_ids = output_ids[0, input_length:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        if not prompts:
            return ()
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts cannot contain an empty prompt.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        model_inputs = self._prepare_batch_inputs(prompts)
        input_length = model_inputs["input_ids"].shape[-1]
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            generate_kwargs.update(temperature=temperature, top_p=top_p)

        with self._torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        completions = output_ids[:, input_length:]
        return tuple(
            text.strip()
            for text in self.tokenizer.batch_decode(
                completions,
                skip_special_tokens=True,
            )
        )

    def _prepare_inputs(self, prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")

        target_device = self._input_device()
        return {name: tensor.to(target_device) for name, tensor in encoded.items()}

    def _prepare_batch_inputs(self, prompts: Sequence[str]) -> dict[str, Any]:
        if getattr(self.tokenizer, "chat_template", None):
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in prompts
            ]
            encoded = self.tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                list(prompts),
                padding=True,
                return_tensors="pt",
            )
        target_device = self._input_device()
        return {name: tensor.to(target_device) for name, tensor in encoded.items()}

    def _input_device(self) -> Any:
        if self.device == "cuda":
            return self.model.device
        return self._torch.device(self.device)

    def _set_seed(self, seed: int) -> None:
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(torch_module.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
    return requested


def _preferred_dtype(device: str, torch_module: Any) -> Any:
    if device == "cuda":
        if getattr(torch_module.cuda, "is_bf16_supported", lambda: False)():
            return torch_module.bfloat16
        return torch_module.float16
    if device == "mps":
        return torch_module.float16
    return torch_module.float32
