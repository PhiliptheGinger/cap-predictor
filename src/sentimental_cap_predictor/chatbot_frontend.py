"""Simple interactive frontend for Qwen chat model using CMD protocol."""

from __future__ import annotations

from pathlib import Path

import requests

# Heavy dependencies are imported lazily in ``main`` to keep the module light
# for unit tests and simple command handling. ``colorama`` is a lightweight
# dependency used to provide coloured prompts for a nicer CLI experience.
from colorama import Fore, Style, init
from sentimental_cap_predictor.data.news import (
    fetch_first_gdelt_article as _fetch_first_gdelt_article,
)


_MEMORY_INDEX = Path("data/memory.faiss")

# Initialise colour handling for cross-platform compatibility
init(autoreset=True)


def fetch_first_gdelt_article(query: str) -> str:
    """Return text for the first GDELT article matching ``query``.

    The helper requests full article content from
    :func:`sentimental_cap_predictor.data.news.fetch_first_gdelt_article` and
    falls back to the headline and URL if content extraction fails.
    """

    try:
        article = _fetch_first_gdelt_article(query, prefer_content=True)
    except requests.RequestException as exc:  # pragma: no cover - network error
        return f"GDELT request failed: {exc}"

    if article.content:
        from sentimental_cap_predictor.memory_indexer import TextMemory

        index_path = _MEMORY_INDEX
        if index_path.exists():
            memory = TextMemory.load(index_path)
        else:
            memory = TextMemory()
        memory.add([article.content])
        memory.save(index_path)
        return article.content
    if article.title and article.url:
        return f"{article.title} - {article.url}"
    return article.title or article.url


SYSTEM_PROMPT = (
    "You are a command planner for a terminal application.\n"
    "Output must be a single line: either\n"
    "CMD: <command> to run, or one clarifying question.\n"
    "Do not include code fences, explanations, or extra lines.\n"
    "Available tools:\n"
    '  • curl "<url>" (defaults: -sSL)\n'
    '  • gdelt search --query "<q>" --limit <n> (default --limit 10)\n'
    "Before responding, self-check that your output abides by these rules."
)


def handle_command(command: str) -> str:
    """Execute a shell ``command`` or route GDELT/news requests.

    When the command mentions ``gdelt`` or ``news`` the GDELT API is queried
    using :func:`fetch_first_gdelt_article` to return article text or a
    headline.  All other commands are executed via the system shell and the
    resulting standard output (or standard error) is returned.
    """

    import re
    import shlex
    import subprocess

    lower = command.lower()
    if "gdelt" in lower or "news" in lower:
        parts = shlex.split(command)
        query: str | None = None
        for i, part in enumerate(parts):
            if part.startswith("--query="):
                query = part.split("=", 1)[1]
                break
            if part in {"--query", "query"} and i + 1 < len(parts):
                query = parts[i + 1]
                break
            if "query=" in part:
                match = re.search(r"query=([^&]+)", part)
                if match:
                    query = match.group(1)
                    break
        if query is None and parts:
            query = parts[-1]
        result = fetch_first_gdelt_article(query)
        if not result:
            return "No news found."
        return result

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    """Run a REPL-style chat session with the local Qwen model."""
    from sentimental_cap_predictor.cmd_utils import extract_cmd
    from sentimental_cap_predictor.config_llm import get_llm_config
    from sentimental_cap_predictor.llm_providers.qwen_local import (
        QwenLocalProvider,
    )

    config = get_llm_config()
    provider = QwenLocalProvider(
        model_path=config.model_path, temperature=config.temperature
    )
    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # ``pending_cmd`` stores a command produced by the model that has not yet
    # been executed.  This avoids ``UnboundLocalError`` when the loop checks
    # the variable before the model has suggested any command.
    pending_cmd: str | None = None

    def run_command(cmd: str) -> str:
        """Execute ``cmd`` via :func:`handle_command` and show the output."""

        output = handle_command(cmd)
        print(output)
        return output

    while True:
        # Execute any command left over from the previous iteration before
        # prompting the user again.
        if pending_cmd:
            output = run_command(pending_cmd)
            history.append({"role": "assistant", "content": output})
            pending_cmd = None
            continue

        try:
            user = input(f"{Fore.CYAN}user>{Style.RESET_ALL} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user})
        reply = provider.chat(history)
        command, question = extract_cmd(reply)
        if command:
            pending_cmd = command
            continue
        if question:
            print(question)
            history.append({"role": "assistant", "content": question})
            continue

        # Retry once with a reminder about the expected format
        history.append(
            {
                "role": "user",
                "content": "Output invalid. Remember the CMD contract.",
            }
        )
        reply = provider.chat(history)
        command, question = extract_cmd(reply)
        if command:
            pending_cmd = command
        elif question:
            print(question)
            history.append({"role": "assistant", "content": question})
        else:
            print(reply)
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
