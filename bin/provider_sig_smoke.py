#!/usr/bin/env python
"""Print provider constructor signatures and validate config bindings."""
from inspect import signature

from sentimental_cap_predictor.llm_core.llm_providers.qwen_local import QwenLocalProvider
from sentimental_cap_predictor.llm_core.llm_providers.deepseek import DeepSeekProvider
from sentimental_cap_predictor.llm_core.provider_config import (
    DeepSeekConfig,
    QwenLocalConfig,
)


def main() -> None:
    providers = [
        ("QwenLocalProvider", QwenLocalProvider, QwenLocalConfig()),
        ("DeepSeekProvider", DeepSeekProvider, DeepSeekConfig()),
    ]
    for name, cls, cfg in providers:
        sig = signature(cls.__init__)
        sig.bind_partial(None, **cfg.model_dump())
        print(name, sig)


if __name__ == "__main__":
    main()
