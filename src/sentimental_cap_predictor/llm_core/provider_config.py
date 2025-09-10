"""Typed configuration models for LLM providers."""
from __future__ import annotations

import os
from pydantic import BaseModel, Field


class QwenLocalConfig(BaseModel):
    """Configuration options for :class:`QwenLocalProvider`."""

    temperature: float = Field(default=0.7)
    max_new_tokens: int = Field(default=512)

    @classmethod
    def from_env(cls) -> "QwenLocalConfig":
        """Load configuration from environment variables."""
        return cls(
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", 512)),
        )
