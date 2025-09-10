#!/usr/bin/env python
"""Print provider constructor signatures and validate config bindings."""
from inspect import signature

from sentimental_cap_predictor.llm_core.llm_providers.qwen_local import QwenLocalProvider
from sentimental_cap_predictor.llm_core.provider_config import QwenLocalConfig


def main() -> None:
    cfg = QwenLocalConfig()
    sig = signature(QwenLocalProvider.__init__)
    sig.bind_partial(None, **cfg.model_dump())
    print("QwenLocalProvider", sig)


if __name__ == "__main__":
    main()
