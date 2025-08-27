"""Qwen provider using the OpenAI client."""

from __future__ import annotations

from typing import Any

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - import-time failure
    raise RuntimeError("OpenAI client is required") from exc

from .base import ChatMessage, LLMProvider


class QwenProvider(LLMProvider):
    """Implementation of :class:`LLMProvider` for the Qwen API."""

    def __init__(
        self, api_key: str, model: str, base_url: str, temperature: float
    ) -> None:
        self.model = model
        self.temperature = temperature
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Send messages to the Qwen model."""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **kwargs,
        )
        return completion.choices[0].message.get("content", "")
