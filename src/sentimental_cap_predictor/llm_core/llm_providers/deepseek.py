"""DeepSeek provider supporting local or API-hosted models."""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from ..utils import clean_generate_kwargs
from .base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


class DeepSeekProvider(LLMProvider):
    """Implementation of :class:`LLMProvider` for DeepSeek models.

    The provider can operate in two modes:

    * **API mode** when an API key is supplied via ``api_key`` or the
      ``DEEPSEEK_API_KEY`` environment variable.  Requests are sent using the
      OpenAI-compatible client.
    * **Local mode** when ``model_path`` (or ``DEEPSEEK_MODEL_PATH``) points to a
      local Hugging Face model checkpoint.  The model is loaded with
      :mod:`transformers` and executed locally.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(f"Unknown args: {sorted(kwargs.keys())}")

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model_id = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.model_path = model_path or os.getenv("DEEPSEEK_MODEL_PATH")

        if self.api_key:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "openai package required for DeepSeek API usage"
                ) from exc
            base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            logger.info("DeepSeek API client initialized model=%s base=%s", self.model_id, base_url)
            self._local = False
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch

            path = self.model_path or self.model_id
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path).eval()
            self._torch = torch
            self.model_id = path
            self._local = True
            logger.info("Loaded DeepSeek local model %s", path)

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        if self._local:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            inputs = tokenized.to(self.model.device)
            kwargs.setdefault("max_new_tokens", self.max_new_tokens)
            kwargs.setdefault(
                "max_length",
                inputs["input_ids"].shape[-1] + kwargs["max_new_tokens"],
            )
            generate_kwargs = clean_generate_kwargs(
                temperature=self.temperature,
                **kwargs,
            )
            with self._torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            start = inputs["input_ids"].shape[-1]
            generated = outputs[0, start:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
        )
        message = completion.choices[0].message
        if isinstance(message, dict):  # compatibility with simple mocks
            return message.get("content", "")
        return getattr(message, "content", "")
