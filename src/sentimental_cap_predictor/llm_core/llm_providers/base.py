"""Common types and protocols for LLM providers."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

ChatMessage = Dict[str, str]
"""Minimal chat message containing ``role`` and ``content`` fields."""


class LLMProvider(Protocol):
    """Protocol for chat-based LLM provider implementations."""

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        """Return the model's response to a list of chat messages."""
