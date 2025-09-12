"""Helpers for fetching and storing news articles from GDELT.

This module wraps :class:`~sentimental_cap_predictor.news.gdelt_client.GdeltClient`
for querying the GDELT API and uses :class:`~sentimental_cap_predictor.news.fetcher.HtmlFetcher`
and :class:`~sentimental_cap_predictor.news.extractor.ArticleExtractor` for
retrieving and processing article content.  Domain allow/block lists are read
from the ``NEWS_ALLOWED_DOMAINS`` and ``NEWS_BLOCKED_DOMAINS`` environment
variables (comma separated) so that policies can be configured without
modifying the code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import textwrap
from urllib.parse import urlparse

from sentimental_cap_predictor.memory import vector_store
from .gdelt_client import GdeltClient
from .fetcher import HtmlFetcher
from .extractor import ArticleExtractor


logger = logging.getLogger(__name__)

_EMPTY_HTML = {
    "<html></html>",
    "<html><head></head><body></body></html>",
    "",
}


def _is_empty_page(html: str) -> bool:
    """Return ``True`` for trivially empty HTML pages."""

    return "".join(html.split()).lower() in _EMPTY_HTML


def _env_list(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(filter(None, (p.strip() for p in value.split(","))))


def domain_blocked(url: str) -> bool:
    domain = urlparse(url).netloc
    blocked = _env_list("NEWS_BLOCKED_DOMAINS")
    return any(d in domain for d in blocked)


def domain_ok(url: str) -> bool:
    allowed = _env_list("NEWS_ALLOWED_DOMAINS")
    if not allowed:
        return True
    domain = urlparse(url).netloc
    return any(d in domain for d in allowed)


async def _fetch_html_async(url: str) -> str:
    fetcher = HtmlFetcher()
    try:
        html = await fetcher.get(url)
    finally:
        await fetcher.aclose()
    return html or ""


def fetch_html(url: str) -> str:
    """Synchronously fetch ``url`` using :class:`HtmlFetcher`."""

    return asyncio.run(_fetch_html_async(url))


def extract_main_text(html: str, url: str | None = None) -> str:
    extractor = ArticleExtractor()
    art = extractor.extract(html, url=url)
    return art.text if art else ""


def search_gdelt(query: str, max_records: int = 15) -> list[dict]:
    """Return article dictionaries for ``query``.

    Results are filtered according to the configured domain policy and pages
    that fail to fetch or are empty are skipped.
    """

    client = GdeltClient()
    try:
        stubs = client.search(query, max_records=max_records)
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("GDELT API request failed: %s", exc)
        return []

    articles: list[dict] = []
    for stub in stubs:
        url = stub.url
        if domain_blocked(url):
            logger.info("Skipping %s: blocked domain", stub.domain)
            continue
        if not domain_ok(url):
            logger.info("Skipping %s: undesired domain", stub.domain)
            continue
        try:  # pragma: no cover - network failure
            html = fetch_html(url)
        except Exception:
            continue
        if _is_empty_page(html):
            continue
        articles.append(
            {
                "title": getattr(stub, "title", "") or "",
                "url": url,
                "domain": stub.domain,
                "language": "",
                "seendate": stub.seendate.isoformat() if stub.seendate else "",
            }
        )
        if len(articles) >= max_records:
            break
    return articles


def summarize(text: str, max_chars: int = 800) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(text, width=max_chars, placeholder="…")


def _chunk_text(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    """Split *text* into character chunks with small overlaps."""

    if size <= 0:
        return []
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + size)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return chunks


def _store_chunks(result: dict) -> None:
    """Store article text chunks in the vector store.

    Any failures are logged but ignored so that the main flow continues.
    """

    text = result.get("text", "")
    metadata = {
        "title": result.get("title", ""),
        "url": result.get("url", ""),
        "seendate": result.get("seendate", ""),
        "domain": result.get("domain", ""),
    }
    chunks = _chunk_text(text)
    for idx, chunk in enumerate(chunks):
        doc_id = f"{metadata['url']}#{idx}"
        try:  # pragma: no cover - network/db failures
            vector_store.upsert(doc_id, chunk, metadata)
        except Exception as exc:  # pragma: no cover - failures handled
            logger.warning("Vector store upsert failed: %s", exc)


def main() -> None:  # pragma: no cover - CLI convenience
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--max", type=int, default=15)
    args = ap.parse_args()

    arts = search_gdelt(args.query, args.max)
    for art in arts:
        url = art.get("url")
        if not url:
            logger.info("Skipping record with no URL")
            continue

        try:
            html = fetch_html(url)
        except Exception as exc:
            logger.warning("Error fetching %s: %s", url, exc)
            continue
        if _is_empty_page(html):
            logger.info("Skipping %s: empty page", url)
            continue

        text = extract_main_text(html, url=url)
        if not text:
            logger.info("Skipping %s: no extractable text", url)
            continue

        result = {
            "title": art.get("title", ""),
            "url": url,
            "domain": art.get("domain", ""),
            "language": art.get("language", ""),
            "seendate": art.get("seendate", ""),
            "text": text,
            "summary": summarize(text),
        }
        _store_chunks(result)
        print(json.dumps(result, ensure_ascii=False))
        return

    # If nothing worked, at least return an empty envelope
    print(json.dumps({}))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

