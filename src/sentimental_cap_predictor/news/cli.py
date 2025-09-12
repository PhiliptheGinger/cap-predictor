"""Lightweight command line helpers for news ingestion.

This module intentionally avoids importing any heavyweight dependencies so it
can be used in constrained environments and unit tests.  The implementation is
limited to wiring together the small ``news`` utilities such as
``GdeltClient``, ``HtmlFetcher`` and ``ArticleExtractor``.

The CLI provides three primary commands:

``search``
    Query the GDELT API and return a list of article stubs as JSON.

``ingest``
    Fetch and extract articles for a query and persist them using the storage
    module.

``score``
    Compute scores for stored articles using the scoring utilities and output
    the results as JSON.

Two additional commands – ``fetch-gdelt`` and ``read`` – are kept for backwards
compatibility with the existing chatbot tests.  They are thin wrappers around
``search`` and the HTML extraction helpers respectively.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional
from enum import Enum

import typer

from .extractor import ArticleExtractor, ExtractedArticle
from .fetcher import HtmlFetcher
from .gdelt_client import GdeltClient
from . import store
from .scoring import score_news


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
) -> None:
    """Fetch articles for ``query`` and store them via :mod:`news.store`."""

    client = GdeltClient()
    extractor = ArticleExtractor()

    count = 0
    for stub in client.search(query, max_records=max_results):
        meta = _stub_to_dict(stub)
        store.upsert_article(meta)
        html = fetch_html(stub.url)
        if not html:
            store.log_error(stub.url, "fetch", "no html")
            continue
        art: Optional[ExtractedArticle] = extractor.extract(html, url=stub.url)
        if not art or not art.text:
            store.log_error(stub.url, "extract", "no text")
            continue
        store.upsert_content(stub.url, art.text)
        count += 1

    typer.echo(str(count))


@app.command("score")
def score_command() -> None:
    """Score stored articles and print JSON records."""

    import pandas as pd

    with store._connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT a.url,
                   a.seendate AS timestamp,
                   LENGTH(c.text) AS length,
                   COALESCE(c.relevance, 0) AS credibility
            FROM articles a
            JOIN contents c ON a.url = c.url
            """,
            conn,
        )

    if df.empty:
        typer.echo("[]")
        return

    scored = score_news(df)
    typer.echo(scored.to_json(orient="records"))


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

