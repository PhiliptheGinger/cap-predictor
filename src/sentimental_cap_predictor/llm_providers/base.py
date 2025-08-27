"""Base interfaces for large language model providers."""

from __future__ import annotations

from typing import Dict, List, Protocol

ChatMessage = Dict[str, str]
"""A chat message with ``role`` and ``content`` fields."""


class LLMProvider(Protocol):
    """Protocol for chat-based large language model providers."""

    def chat(self, messages: List[ChatMessage], **kwargs: object) -> str:
        """Generate a reply for a sequence of chat ``messages``.

        Additional keyword arguments may be supplied to configure the
        provider. The provider should return the generated text response as a
        string.
        """
        ...
