"""Configuration helpers for LLM providers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_llm_config() -> dict[str, Any]:
    """Return configuration for a local Qwen LLM."""

    model_path = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen2-1.5B-Instruct")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
    return {
        "model_path": model_path,
        "temperature": temperature,
    }
