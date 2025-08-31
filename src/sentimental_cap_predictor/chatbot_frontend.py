"""Simple interactive frontend for Qwen chat model using CMD protocol."""

from __future__ import annotations

import re
import subprocess

# Heavy dependencies are imported lazily in ``main`` to keep the module light
# for unit tests and simple command handling.


def fetch_first_gdelt_article(query: str) -> str:  # pragma: no cover
    """Placeholder for the news lookup helper.

    The real implementation lives in :mod:`sentimental_cap_predictor.dataset`.
    It is patched in tests to avoid network calls.
    """

    return ""

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


def handle_command(command: str) -> str:
    """Execute a shell ``command`` or route GDELT/news requests.

    When the command mentions ``gdelt`` or ``news`` the GDELT API is queried
    using :func:`fetch_first_gdelt_article` to return article text or a
    headline.  All other commands are executed via the system shell and the
    resulting standard output (or standard error) is returned.
    """

    lower = command.lower()
    if "gdelt" in lower or "news" in lower:
        match = re.search(r"query=([^&\s]+)", command)
        query = match.group(1) if match else command.split()[-1]
        return fetch_first_gdelt_article(query) or "No news found."

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    """Run a REPL-style chat session with the local Qwen model."""
    from sentimental_cap_predictor.config_llm import get_llm_config
    from sentimental_cap_predictor.llm_providers.qwen_local import QwenLocalProvider

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
        if reply.startswith("CMD:"):
            output = handle_command(reply[4:].strip())
            print(output)
            history.append({"role": "assistant", "content": output})
        else:
            print(reply)
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
