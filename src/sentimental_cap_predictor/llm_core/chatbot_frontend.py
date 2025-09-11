# flake8: noqa
"""Interactive chat frontend supporting multiple LLM backends and agent mode.

The script can run either a basic REPL or an experimental agent loop capable of
executing registered tools.  Configuration is read from environment variables or
corresponding command line flags; run with ``--help`` for details.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import requests

# Heavy dependencies are imported lazily in ``main`` to keep the module light
# for unit tests and simple command handling. ``colorama`` is a lightweight
# dependency used to provide coloured prompts for a nicer CLI experience.
from colorama import Fore, Style, init

logger = logging.getLogger(__name__)

from sentimental_cap_predictor.data.news import (
    FetchArticleSpec,
)
from sentimental_cap_predictor.data.news import fetch_article as _fetch_article
from sentimental_cap_predictor.llm_core.provider_config import LLMProviderEnum
from sentimental_cap_predictor.reasoning.engine import (
    analogy_explain,
    reason_about,
    simulate,
)

_MEMORY_INDEX: Path | None = None
_SEEN_URLS: set[str] = set()
_SEEN_TITLES: set[str] = set()
# JSON file for persisting seen articles to avoid repeated headlines.
# Stored at ``data/gdelt_seen.json`` as a list of objects with ``title`` and
# ``url`` keys.
_SEEN_META_PATH = Path("data/gdelt_seen.json")
_SEEN_METADATA: list[dict[str, str]] = []
_SEEN_LOADED = False
_LAST_ARTICLE_URL: str | None = None
_ALLOWED_CURL_DOMAINS = {"api.gdeltproject.org"}


def _parse_args() -> argparse.Namespace:
    """Return parsed command line arguments."""

    parser = argparse.ArgumentParser(description="Interactive chat frontend")
    parser.add_argument(
        "--provider",
        choices=[p.value for p in LLMProviderEnum],
        default=os.getenv("LLM_PROVIDER", LLMProviderEnum.qwen_local.value),
        help="LLM backend provider (or set LLM_PROVIDER)",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        default=os.getenv("AGENT_MODE") not in {None, "", "0"},
        help="Enable agent mode with tool execution (AGENT_MODE=1)",
    )
    parser.add_argument(
        "--confirm-cmds",
        action="store_true",
        default=os.getenv("CONFIRM_CMDS") not in {None, "", "0"},
        help="Require confirmation before running tools (CONFIRM_CMDS=1)",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        default=os.getenv("ENABLE_WEB_SEARCH") not in {None, "", "0"},
        help="Register web search tool",
    )
    parser.add_argument(
        "--enable-python",
        action="store_true",
        default=os.getenv("ENABLE_PYTHON_RUN") not in {None, "", "0"},
        help="Register python.run tool",
    )
    parser.add_argument(
        "--enable-file-write",
        action="store_true",
        default=os.getenv("ENABLE_FILE_WRITE") not in {None, "", "0"},
        help="Register file.write tool",
    )
    parser.add_argument(
        "--enable-read-url",
        action="store_true",
        default=os.getenv("ENABLE_READ_URL") not in {None, "", "0"},
        help="Register read_url tool",
    )
    parser.add_argument(
        "--enable-memory",
        action="store_true",
        default=os.getenv("ENABLE_MEMORY") not in {None, "", "0"},
        help="Register memory tools",
    )
    return parser.parse_args()


def _get_provider(name: str):
    """Instantiate the requested LLM provider."""

    if name == LLMProviderEnum.deepseek.value:
        from sentimental_cap_predictor.llm_core.llm_providers.deepseek import (
            DeepSeekProvider,
        )
        from sentimental_cap_predictor.llm_core.provider_config import (
            DeepSeekConfig,
        )

        cfg = DeepSeekConfig.from_env()
        return DeepSeekProvider(**cfg.model_dump())

    from sentimental_cap_predictor.llm_core.llm_providers.qwen_local import (
        QwenLocalProvider,
    )
    from sentimental_cap_predictor.llm_core.provider_config import QwenLocalConfig

    cfg = QwenLocalConfig.from_env()
    return QwenLocalProvider(**cfg.model_dump())


def _load_agent_tools(args: argparse.Namespace) -> None:
    """Import optional agent tools based on command line flags."""

    if args.enable_python:
        import tools.python_exec  # noqa: F401
    if args.enable_file_write:
        import tools.file_io  # noqa: F401
    if args.enable_read_url:
        import tools.read_url  # noqa: F401
    if args.enable_memory:
        import tools.memory  # noqa: F401
    if args.enable_web_search:
        from sentimental_cap_predictor.llm_core.agent import tools as _  # noqa: F401


def _load_seen_metadata() -> None:
    """Populate seen URL/title sets from _SEEN_META_PATH if available."""
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
    except (
        requests.RequestException,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover
        if isinstance(exc, requests.RequestException):
            return f"GDELT request failed: {exc}"
        return "No readable article found"

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
    "CMD: <command> to run, a concise natural-language answer when no command is needed, or one clarifying question.\n"
    "Do not include code fences, explanations, or extra lines.\n"
    "Available tools:\n"
    '  • curl "<url>" (defaults: -sSL)\n'
    '  • gdelt search --query "<q>" --limit <n> '
    "(default --limit 10)\n"
    "  • article.summarize_last (summarize last fetched article)\n"
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
    import urllib.parse

    if _MEMORY_INDEX is None:
        setup()

    global _LAST_ARTICLE_URL

    lower = command.lower()
    if lower.strip() == "article.summarize_last":
        if not _LAST_ARTICLE_URL:
            return "No article available for summarization."
        return handle_command(f"news.read --url {_LAST_ARTICLE_URL} --summarize")
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
        if url:
            _LAST_ARTICLE_URL = url
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
        result = fetch_first_gdelt_article(stripped)
        if _SEEN_METADATA:
            _LAST_ARTICLE_URL = _SEEN_METADATA[-1].get("url")
        return result

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
        if _SEEN_METADATA:
            _LAST_ARTICLE_URL = _SEEN_METADATA[-1].get("url")
        if not result:
            return "No news found."
        return result

    tokens = shlex.split(command)
    if tokens and tokens[0] == "curl":
        urls = [
            t for t in tokens[1:] if t.startswith("http://") or t.startswith("https://")
        ]
        for url in urls:
            domain = urllib.parse.urlparse(url).hostname or ""
            if domain == "example.com" or domain not in _ALLOWED_CURL_DOMAINS:
                return f"Access to {domain} is not permitted."

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or result.stderr.strip()


def _route_keywords(message: str) -> Callable[[], str] | None:
    """Return a callable handling ``message`` if it matches known patterns."""
    import re

    m = re.search(
        r"(?:pull up|fetch|get) (?:an? )?article(?: (?:about|on) (?P<topic>.+))?",
        message,
        re.I,
    )
    if m:
        topic = m.group("topic")
        if topic:
            topic = topic.strip().rstrip("?.! ")

            def _fetch():
                result = fetch_first_gdelt_article(topic)
                global _LAST_ARTICLE_URL
                if _SEEN_METADATA:
                    _LAST_ARTICLE_URL = _SEEN_METADATA[-1].get("url")
                return result

            return _fetch

        def _ask_topic() -> str:
            return "What topic should I search for?"

        return _ask_topic

    if re.search(r"\bread(?: it| (?:that|the) article)\b", message, re.I):

        def _read():
            if not _LAST_ARTICLE_URL:
                return "No article available for reading."
            return handle_command(f"news.read --url {_LAST_ARTICLE_URL}")

        return _read

    if re.search(
        r"\bsummarize(?: it| (?:that|the) article)\b|tell me what['’]s on this page",
        message,
        re.I,
    ):

        def _summarize():
            return handle_command("article.summarize_last")

        return _summarize

    if re.search(r"what did you load\?", message, re.I):

        def _last_loaded():
            if _SEEN_METADATA:
                last = _SEEN_METADATA[-1]
                title = last.get("title")
                url = last.get("url")
                if title and url:
                    return f"{title} - {url}"
                return title or url or "Nothing loaded yet."
            return "Nothing loaded yet."

        return _last_loaded

    m = re.search(r"search memory for (.+)", message, re.I)
    if m:
        query = m.group(1).strip()

        def _search():
            return handle_command(f'memory search "{query}"')

        return _search

    m = re.search(r"reason about (.+)", message, re.I)
    if m:
        topic = m.group(1).strip()

        def _reason():
            logger.info("Dispatching reason_about: %s", topic)
            result = reason_about(topic)
            logger.info("Finished reason_about: %s", topic)
            return result

        return _reason

    m = re.search(r"simulate (.+)", message, re.I)
    if m:
        scenario = m.group(1).strip()

        def _simulate():
            logger.info("Dispatching simulate: %s", scenario)
            result = simulate(scenario)
            logger.info("Finished simulate: %s", scenario)
            return result

        return _simulate

    m = re.search(r"explain with analogy (.+?) to (.+)", message, re.I)
    if m:
        src = m.group(1).strip()
        tgt = m.group(2).strip()

        def _analogy():
            logger.info("Dispatching analogy_explain: %s -> %s", src, tgt)
            result = analogy_explain(src, tgt)
            logger.info("Finished analogy_explain: %s -> %s", src, tgt)
            return result

        return _analogy

    return None


def _classify_and_route(message: str) -> str | None:
    """Use intent classification to handle simple requests."""

    try:
        from sentimental_cap_predictor.llm_core.chatbot_nlu import qwen_intent
    except Exception:  # pragma: no cover - import failure
        return None

    try:
        text = qwen_intent.call_qwen(message)
    except Exception:  # pragma: no cover - model failure
        return None

    m = qwen_intent._JSON_RE.search(text or "")  # type: ignore[attr-defined]
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

    intent = data.get("intent")
    slots = data.get("slots") or {}
    if intent == "info.lookup":
        query = slots.get("query")
        if not query:
            keywords = slots.get("keywords")
            if isinstance(keywords, list) and keywords:
                query = keywords[0]
        if query:
            return fetch_first_gdelt_article(query)

    return "Try: pull up an article about X."


def main() -> None:
    """Run a REPL-style chat session or agent loop depending on flags."""

    args = _parse_args()
    setup()

    try:
        provider = _get_provider(args.provider)
    except TypeError as exc:  # pragma: no cover - defensive
        print(f"Provider init failed: {exc}")
        return

    if args.confirm_cmds:
        os.environ.setdefault("CONFIRM_CMDS", "1")

    if args.agent:
        _load_agent_tools(args)
        from sentimental_cap_predictor.llm_core.agent import AgentLoop

        loop = AgentLoop(
            lambda prompt: provider.chat([{"role": "user", "content": prompt}])
        )
        while True:
            try:
                user = input(f"{Fore.CYAN}user>{Style.RESET_ALL} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            try:
                reply = loop.run(user)
            except Exception as exc:  # pragma: no cover - runtime errors
                reply = f"Error: {exc}"
            print(reply)
        return

    from sentimental_cap_predictor.cmd_utils import extract_cmd

    history: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    pending_cmd: str | None = None

    def run_command(cmd: str) -> str:
        output = handle_command(cmd)
        print(output)
        return output

    while True:
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

        handler = _route_keywords(user)
        if handler:
            logger.info("Dispatching routed message: %s", user)
            output = handler()
            print(output)
            history.append({"role": "assistant", "content": output})
            logger.info("Finished routed message: %s", user)
            continue
        result = _classify_and_route(user)
        if result:
            print(result)
            history.append({"role": "assistant", "content": result})
            continue

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

        print(reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
