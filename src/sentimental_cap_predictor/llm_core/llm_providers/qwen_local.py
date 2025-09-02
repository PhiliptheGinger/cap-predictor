"""Qwen provider using local transformer weights."""

from __future__ import annotations

from typing import Any, List

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .base import ChatMessage, LLMProvider


class QwenLocalProvider(LLMProvider):
    """Implementation of :class:`LLMProvider` for a local Qwen model."""

    def __init__(
        self, model_path: str, temperature: float, max_new_tokens: int = 512
    ) -> None:
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        self.model = load_checkpoint_and_dispatch(
            model, model_path, device_map="auto"
        )
        self.model.eval()

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Return the model's response to a list of chat messages."""

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        kwargs.setdefault(
            "max_length", inputs["input_ids"].shape[-1] + kwargs["max_new_tokens"]
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, temperature=self.temperature, **kwargs
            )
        generated = outputs[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

