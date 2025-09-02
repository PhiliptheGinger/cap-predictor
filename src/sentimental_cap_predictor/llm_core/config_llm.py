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


def get_llm_config() -> LLMConfig:
    """Return configuration for a local Qwen LLM."""

    model_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2-1.5B-Instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
    return LLMConfig(model_path=model_path, temperature=temperature)
