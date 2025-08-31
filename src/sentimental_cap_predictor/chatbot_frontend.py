"""Simple interactive frontend for Qwen chat model using CMD protocol."""

from __future__ import annotations

from sentimental_cap_predictor.config_llm import get_llm_config
from sentimental_cap_predictor.llm_providers.qwen_local import (
    QwenLocalProvider,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant."
    "\nIf you want me to run a shell command, respond with 'CMD: <command>'."
    "\nFor normal replies, respond without the prefix."
    "\nGDELT articles should be fetched from "
    "https://api.gdeltproject.org/api/v2/doc/doc."
    "\nExample: CMD: curl "
    '"https://api.gdeltproject.org/api/v2/doc/doc?query=ukraine&'
    'mode=ArtList&format=json"'
)


def main() -> None:
    """Run a REPL-style chat session with the local Qwen model."""
    config = get_llm_config()
    provider = QwenLocalProvider(
        model_path=config.model_path, temperature=config.temperature
    )
    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user})
        reply = provider.chat(history)
        print(reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
