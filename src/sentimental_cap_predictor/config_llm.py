"""Configuration helpers for LLM providers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_llm_config() -> dict[str, Any]:
    """Return configuration for Qwen LLM from environment variables."""

    base_url = os.getenv(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = os.getenv("QWEN_API_KEY", "")
    model = os.getenv("QWEN_MODEL", "qwen-max")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.7))
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
    }
