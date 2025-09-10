"""Typed configuration models for LLM providers."""
from __future__ import annotations

import os
from enum import Enum
from pydantic import BaseModel, Field


class LLMProviderEnum(str, Enum):
    """Enumerate supported LLM provider backends."""

    qwen_local = "qwen_local"
    deepseek = "deepseek"


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


DEFAULT_DEEPSEEK_INSTRUCT_MODEL = "deepseek-chat"
"""Default model name for DeepSeek instruct/chat usage."""

DEFAULT_DEEPSEEK_CODER_MODEL = "deepseek-coder"
"""Default model name for DeepSeek code generation."""


class DeepSeekConfig(BaseModel):
    """Configuration options for :class:`DeepSeekProvider`."""

    temperature: float = Field(default=0.7)
    max_new_tokens: int = Field(default=512)
    model: str = Field(default=DEFAULT_DEEPSEEK_INSTRUCT_MODEL)
    model_path: str | None = Field(default=None)
    api_key: str | None = Field(default=None)

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        """Load configuration from environment variables."""
        return cls(
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.7)),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", 512)),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_INSTRUCT_MODEL),
            model_path=os.getenv("DEEPSEEK_MODEL_PATH"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
