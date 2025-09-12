"""Lightweight command line helpers for news ingestion.

This module intentionally avoids importing any heavyweight dependencies so it
can be used in constrained environments and unit tests.  The implementation is
limited to wiring together the small ``news`` utilities such as
``GdeltClient``, ``HtmlFetcher`` and ``ArticleExtractor``.

The CLI provides three primary commands:

``search``
    Query the GDELT API and return a list of article stubs as JSON.

``ingest``
    Fetch and extract articles for a query and store them in a local JSONL
    file.  Each line contains a JSON object with the article's metadata and
    extracted text.

``list``
    Print the stored articles from the JSONL file.

Two additional commands – ``fetch-gdelt`` and ``read`` – are kept for backwards
compatibility with the existing chatbot tests.  They are thin wrappers around
``search`` and the HTML extraction helpers respectively.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterable, Optional
from enum import Enum

import typer

from .extractor import ArticleExtractor, ExtractedArticle
from .fetcher import HtmlFetcher
from .gdelt_client import GdeltClient


app = typer.Typer(help="Utilities for working with news articles")


# ---------------------------------------------------------------------------
# Helper functions


async def _fetch_html_async(url: str) -> str:
    """Return HTML for ``url`` using :class:`HtmlFetcher`."""

    fetcher = HtmlFetcher()
    try:
        html = await fetcher.get(url)
    finally:
        await fetcher.aclose()
    return html or ""


def fetch_html(url: str) -> str:
    """Synchronously fetch ``url`` using the asynchronous fetcher."""

    return asyncio.run(_fetch_html_async(url))


def _stub_to_dict(stub) -> dict[str, object]:  # pragma: no cover - tiny helper
    return {
        "title": getattr(stub, "title", "") or "",
        "url": stub.url,
        "domain": getattr(stub, "domain", ""),
        "seendate": stub.seendate.isoformat() if stub.seendate else "",
        "source": getattr(stub, "source", None),
    }


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:  # pragma: no cover - malformed file
            continue
    return items


# ---------------------------------------------------------------------------
# Core commands


@app.command("search")
def search_command(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    max_results: int = typer.Option(3, "--max", "-m", help="Maximum results"),
) -> None:
    """Search the GDELT API and return article stubs as JSON."""

    client = GdeltClient()
    stubs = client.search(query, max_records=max_results)
    typer.echo(json.dumps([_stub_to_dict(s) for s in stubs]))


@app.command("ingest")
def ingest_command(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    max_results: int = typer.Option(3, "--max", "-m", help="Maximum results"),
    store: Path = typer.Option(
        Path("data/news.jsonl"),
        "--store",
        "-s",
        help="File for storing extracted articles",
    ),
) -> None:
    """Fetch and store articles for ``query`` to ``store`` as JSONL."""

    client = GdeltClient()
    extractor = ArticleExtractor()

    records: list[dict] = []
    for stub in client.search(query, max_records=max_results):
        html = fetch_html(stub.url)
        if not html:
            continue
        art: Optional[ExtractedArticle] = extractor.extract(html, url=stub.url)
        if not art or not art.text:
            continue
        rec = _stub_to_dict(stub)
        rec["text"] = art.text
        if art.title and not rec["title"]:
            rec["title"] = art.title
        rec["byline"] = art.byline
        rec["date"] = art.date
        records.append(rec)

    _write_jsonl(store, records)
    typer.echo(str(len(records)))


@app.command("list")
def list_command(
    store: Path = typer.Option(
        Path("data/news.jsonl"),
        "--store",
        "-s",
        help="File previously used with ingest",
    ),
) -> None:
    """Print stored article records as JSON."""

    items = _load_jsonl(store)
    typer.echo(json.dumps(items))


# ---------------------------------------------------------------------------
# Compatibility wrappers


@app.command("fetch-gdelt")
def fetch_gdelt_command(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    max_results: int = typer.Option(3, "--max", "-m", help="Maximum results"),
) -> None:
    """Backward compatible wrapper for :func:`search_command`."""

    search_command(query=query, max_results=max_results)


class TranslateMode(str, Enum):  # pragma: no cover - backwards compatibility
    off = "off"
    auto = "auto"
    en = "en"


@app.command("read")
def read_command(  # pragma: no cover - simple passthrough used in tests
    url: str = typer.Option(..., "--url", "-u", help="Article URL"),
    summarize: bool = typer.Option(False, "--summarize"),
    analyze: bool = typer.Option(False, "--analyze"),
    chunks: Optional[int] = typer.Option(None, "--chunks"),
    overlap: int = typer.Option(0, "--overlap"),
    translate: TranslateMode = typer.Option(TranslateMode.off, "--translate"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Fetch and print article text for ``url``.

    The additional parameters are accepted for backwards compatibility but are
    currently ignored to keep the implementation lightweight.
    """

    html = fetch_html(url)
    extractor = ArticleExtractor()
    art = extractor.extract(html, url=url)
    text = art.text if art else ""
    if json_output:
        typer.echo(json.dumps({"text": text}))
    else:
        typer.echo(text)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()

