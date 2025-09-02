"""Simple interactive frontend for Qwen chat model using CMD protocol."""

from __future__ import annotations

from pathlib import Path

import json
import requests

# Heavy dependencies are imported lazily in ``main`` to keep the module light
# for unit tests and simple command handling. ``colorama`` is a lightweight
# dependency used to provide coloured prompts for a nicer CLI experience.
from colorama import Fore, Style, init

from sentimental_cap_predictor.data.news import (
    FetchArticleSpec,
)
from sentimental_cap_predictor.data.news import fetch_article as _fetch_article

_MEMORY_INDEX: Path | None = None
_SEEN_URLS: set[str] = set()
_SEEN_TITLES: set[str] = set()
# JSON file for persisting seen articles to avoid repeated headlines.
# Stored at ``data/gdelt_seen.json`` as a list of objects with ``title`` and ``url`` keys.
_SEEN_META_PATH = Path("data/gdelt_seen.json")
_SEEN_METADATA: list[dict[str, str]] = []
_SEEN_LOADED = False


def _load_seen_metadata() -> None:
    """Populate seen URL/title sets from :data:`_SEEN_META_PATH` if available."""
    global _SEEN_LOADED
    if _SEEN_LOADED:
        return
    _SEEN_LOADED = True
    if _SEEN_META_PATH.exists():
        try:
            _SEEN_METADATA[:] = json.loads(_SEEN_META_PATH.read_text())
        except json.JSONDecodeError:
            _SEEN_METADATA.clear()
        for item in _SEEN_METADATA:
            if url := item.get("url"):
                _SEEN_URLS.add(url)
            if title := item.get("title"):
                _SEEN_TITLES.add(title)


def setup(memory_index: Path | None = None) -> None:
    """Configure colour handling and default paths."""
    init(autoreset=True)
    global _MEMORY_INDEX
    if _MEMORY_INDEX is None:
        _MEMORY_INDEX = memory_index or Path("data/memory.faiss")


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
    query: str,
    *,
    days: int = 1,
    limit: int = 100,
) -> str:
    """Return text for the first GDELT article matching ``query``.

    The helper requests full article content from
    :func:`sentimental_cap_predictor.data.news.fetch_first_gdelt_article` and
    falls back to the headline and URL if content extraction fails.
    """

    if _MEMORY_INDEX is None:
        setup()
    _load_seen_metadata()

    try:
        article = _fetch_first_gdelt_article(
            query, prefer_content=True, days=days, max_records=limit
        )
    except requests.RequestException as exc:  # pragma: no cover
        return f"GDELT request failed: {exc}"

    if article.title or article.url:
        entry = {"title": article.title, "url": article.url}
        if entry not in _SEEN_METADATA:
            _SEEN_METADATA.append(entry)
            _SEEN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
            _SEEN_META_PATH.write_text(json.dumps(_SEEN_METADATA))

    if article.content:
        from sentimental_cap_predictor.llm_core.memory_indexer import (
            TextMemory,
        )

        index_path = _MEMORY_INDEX
        if index_path.exists():
            memory = TextMemory.load(index_path)
        else:
            memory = TextMemory()
        memory.add([article.content])
        memory.save(index_path)

        meta_path = index_path.with_suffix(".json")

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

    if _MEMORY_INDEX is None:
        setup()

    lower = command.lower()
    if lower.startswith("memory search"):
        from sentimental_cap_predictor.llm_core.memory_indexer import (
            TextMemory,
        )

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

    if lower.startswith("news.fetch_gdelt"):
        import contextlib
        import io

        from sentimental_cap_predictor.news.cli import fetch_gdelt_command

        parts = shlex.split(command)
        query: str | None = None
        max_results = 3
        for i, part in enumerate(parts[1:], start=1):
            if part in {"--query", "-q"} and i + 1 < len(parts):
                query = parts[i + 1]
            elif part.startswith("--query=") or part.startswith("-q="):
                query = part.split("=", 1)[1]
            elif part in {"--max", "-m"} and i + 1 < len(parts):
                try:
                    max_results = int(parts[i + 1])
                except ValueError:
                    pass
            elif part.startswith("--max=") or part.startswith("-m="):
                try:
                    max_results = int(part.split("=", 1)[1])
                except ValueError:
                    pass
        if query is None and len(parts) > 1:
            query = parts[-1]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fetch_gdelt_command(query=query, max_results=max_results)
        return buf.getvalue().strip()

    if lower.startswith("news.read"):
        import contextlib
        import io

        from sentimental_cap_predictor.news.cli import (
            TranslateMode,
            read_command,
        )

        parts = shlex.split(command)
        url: str | None = None
        summarize = False
        analyze = False
        chunks: int | None = None
        overlap = 0
        translate = TranslateMode.off
        i = 1
        while i < len(parts):
            part = parts[i]
            if part in {"--url", "-u"} and i + 1 < len(parts):
                url = parts[i + 1]
                i += 2
                continue
            if part.startswith("--url=") or part.startswith("-u="):
                url = part.split("=", 1)[1]
                i += 1
                continue
            if part == "--summarize":
                summarize = True
                i += 1
                continue
            if part == "--analyze":
                analyze = True
                i += 1
                continue
            if part.startswith("--chunks="):
                try:
                    chunks = int(part.split("=", 1)[1])
                except ValueError:
                    pass
                i += 1
                continue
            if part == "--chunks" and i + 1 < len(parts):
                try:
                    chunks = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if part.startswith("--overlap="):
                try:
                    overlap = int(part.split("=", 1)[1])
                except ValueError:
                    pass
                i += 1
                continue
            if part == "--overlap" and i + 1 < len(parts):
                try:
                    overlap = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if part.startswith("--translate="):
                try:
                    translate = TranslateMode(part.split("=", 1)[1])
                except ValueError:
                    pass
                i += 1
                continue
            if part == "--translate" and i + 1 < len(parts):
                try:
                    translate = TranslateMode(parts[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            i += 1
        if url is None and len(parts) > 1:
            url = parts[-1]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            read_command(
                url=url,
                summarize=summarize,
                analyze=analyze,
                chunks=chunks,
                overlap=overlap,
                translate=translate,
            )
        return buf.getvalue().strip()

    # Treat single bare words without a matching executable as GDELT queries.
    import shutil

    stripped = command.strip()
    if " " not in stripped and not shutil.which(stripped):
        return fetch_first_gdelt_article(stripped)

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
    setup()
    from sentimental_cap_predictor.cmd_utils import extract_cmd
    from sentimental_cap_predictor.llm_core.config_llm import get_llm_config
    from sentimental_cap_predictor.llm_core.llm_providers.qwen_local import (
        QwenLocalProvider,
    )

    config = get_llm_config()
    provider = QwenLocalProvider(
        model_path=config.model_path,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
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
