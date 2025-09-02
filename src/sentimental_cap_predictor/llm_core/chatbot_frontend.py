"""Simple interactive frontend for Qwen chat model using CMD protocol."""

from __future__ import annotations

from pathlib import Path

import requests

# Heavy dependencies are imported lazily in ``main`` to keep the module light
# for unit tests and simple command handling. ``colorama`` is a lightweight
# dependency used to provide coloured prompts for a nicer CLI experience.
from colorama import Fore, Style, init

from sentimental_cap_predictor.data.news import (
    fetch_article as _fetch_article,
    FetchArticleSpec,
)

_MEMORY_INDEX = Path("data/memory.faiss")

_SEEN_URLS: set[str] = set()
_SEEN_TITLES: set[str] = set()

# Initialise colour handling for cross-platform compatibility
init(autoreset=True)


def _fetch_first_gdelt_article(
    query: str,
    *,
    prefer_content: bool = True,
    days: int = 1,
    max_records: int = 100,
):
    spec = FetchArticleSpec(
        query=query,
        days=days,
        max_records=max_records,
        require_text_accessible=prefer_content,
        novelty_against_urls=tuple(_SEEN_URLS),
    )
    article = _fetch_article(spec, seen_titles=_SEEN_TITLES)
    if article.url:
        _SEEN_URLS.add(article.url)
    if article.title:
        _SEEN_TITLES.add(article.title)
    return article


def fetch_first_gdelt_article(
    query: str, *, days: int = 1, limit: int = 100
) -> str:
    """Return text for the first GDELT article matching ``query``.

    The helper requests full article content from
    :func:`sentimental_cap_predictor.data.news.fetch_first_gdelt_article` and
    falls back to the headline and URL if content extraction fails.
    """

    try:
        article = _fetch_first_gdelt_article(
            query, prefer_content=True, days=days, max_records=limit
        )
    except requests.RequestException as exc:  # pragma: no cover
        return f"GDELT request failed: {exc}"

    if article.content:
        from sentimental_cap_predictor.llm_core.memory_indexer import TextMemory

        index_path = _MEMORY_INDEX
        if index_path.exists():
            memory = TextMemory.load(index_path)
        else:
            memory = TextMemory()
        memory.add([article.content])
        memory.save(index_path)

        meta_path = index_path.with_suffix(".json")
        import json

        metadata: list[dict[str, str]]
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                metadata = []
        else:
            metadata = []
        metadata.append({"title": article.title, "url": article.url})
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata))

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
    '  • gdelt search --query "<q>" --limit <n> '
    "(default --limit 10)\n"
    "Before responding, self-check that your output abides by these rules."
)


def handle_command(command: str) -> str:
    """Execute a shell ``command`` or route GDELT/news requests.

    When the command mentions ``gdelt`` or ``news`` the GDELT API is queried
    using :func:`fetch_first_gdelt_article` to return article text or a
    headline.  All other commands are executed via the system shell and the
    resulting standard output (or standard error) is returned.
    """

    import json
    import re
    import shlex
    import subprocess

    lower = command.lower()
    if lower.startswith("memory search"):
        from sentimental_cap_predictor.llm_core.memory_indexer import TextMemory

        index_path = _MEMORY_INDEX
        if not index_path.exists():
            return "No memory index found."

        memory = TextMemory.load(index_path)
        meta_path = index_path.with_suffix(".json")
        if not meta_path.exists():
            return "No memory metadata found."
        metadata = json.loads(meta_path.read_text())

        query = command.removeprefix("memory search").strip()
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        embedding = memory.embed([query])
        distances, indices = memory.index.search(
            embedding,
            min(5, len(metadata)),
        )
        results: list[str] = []
        for idx in indices[0]:
            if idx < len(metadata):
                doc = metadata[idx]
                title = doc.get("title")
                url = doc.get("url")
                if title and url:
                    results.append(f"{title} - {url}")
        return "\n".join(results) if results else "No matches found."

    if "gdelt" in lower or "news" in lower:
        parts = shlex.split(command)
        query: str | None = None
        days: int | None = None
        limit: int | None = None
        for i, part in enumerate(parts):
            if part.startswith("--query="):
                query = part.split("=", 1)[1]
            elif part in {"--query", "query"} and i + 1 < len(parts):
                query = parts[i + 1]
            elif "query=" in part:
                match = re.search(r"query=([^&]+)", part)
                if match:
                    query = match.group(1)
            elif part.startswith("--days="):
                try:
                    days = int(part.split("=", 1)[1])
                except ValueError:
                    pass
            elif part == "--days" and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    pass
            elif part.startswith("--limit="):
                try:
                    limit = int(part.split("=", 1)[1])
                except ValueError:
                    pass
            elif part in {"--limit", "limit"} and i + 1 < len(parts):
                try:
                    limit = int(parts[i + 1])
                except ValueError:
                    pass
        if query is None and parts:
            query = parts[-1]
        kwargs = {}
        if days is not None:
            kwargs["days"] = days
        if limit is not None:
            kwargs["limit"] = limit
        result = fetch_first_gdelt_article(query, **kwargs)
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
    from sentimental_cap_predictor.llm_core.config_llm import get_llm_config
    from sentimental_cap_predictor.llm_core.llm_providers.qwen_local import (
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
