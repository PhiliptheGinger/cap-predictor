"""Configuration helpers for LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration values for a local Qwen model."""

    model_path: str
    temperature: float
    max_new_tokens: int = 512
    # Optional directory used by ``accelerate`` to offload model weights. If
    # ``None`` a directory named "offload" inside the model checkpoint is used
    # automatically.
    offload_folder: str | None = None


def get_llm_config() -> LLMConfig:
    """Return configuration for a local Qwen LLM."""

    model_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2-1.5B-Instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
    max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", 512))
    offload_folder = os.getenv("QWEN_OFFLOAD_FOLDER")
    return LLMConfig(
        model_path=model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        offload_folder=offload_folder,
    )
