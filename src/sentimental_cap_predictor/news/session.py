from __future__ import annotations

"""Session-level helpers for working with fetched news articles."""

import logging
import sys
from typing import List

import requests

from sentimental_cap_predictor.memory import vector_store
from sentimental_cap_predictor.memory.session_state import STATE


logger = logging.getLogger(__name__)


def _get_fetch_gdelt():
    mod = sys.modules.get("sentimental_cap_predictor.news.fetch_gdelt")
    if mod is None:  # pragma: no cover - fallback
        from . import fetch_gdelt as mod  # type: ignore
    return mod


def handle_fetch(topic: str) -> str:
    """Fetch the latest article about *topic* and store it in session state.

    The function queries the GDELT API via :mod:`fetch_gdelt`, extracts the
    article text, stores text chunks in the vector store and records the
    article in :data:`STATE`. A short message describing the loaded article
    is returned. If no suitable article is found an explanatory message is
    returned instead.
    """

    fetch_gdelt = _get_fetch_gdelt()
    try:
        articles = fetch_gdelt.search_gdelt(topic, max_records=15)
    except requests.RequestException as exc:  # pragma: no cover - network failure
        logger.warning("GDELT search failed for %s: %s", topic, exc)
        articles = []
    for art in articles:
        url = art.get("url")
        if not url:
            continue
        if fetch_gdelt.domain_blocked(url) or not fetch_gdelt.domain_ok(url):
            continue
        try:
            html = fetch_gdelt.fetch_html(url)
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.warning("Network error fetching %s: %s", url, exc)
            continue
        except Exception as exc:  # pragma: no cover - unexpected failure
            logger.warning("Error fetching %s: %s", url, exc)
            continue
        if fetch_gdelt._is_empty_page(html):
            logger.info("Skipping %s: empty page", url)
            continue
        try:
            text = fetch_gdelt.extract_main_text(html, url=url)
        except Exception as exc:  # pragma: no cover - extraction failure
            logger.warning("Error processing %s: %s", url, exc)
            continue
        if not text:
            continue
        result = {
            "title": art.get("title", ""),
            "url": url,
            "domain": art.get("domain", ""),
            "language": art.get("language", ""),
            "seendate": art.get("seendate", ""),
            "text": text,
            "summary": fetch_gdelt.summarize(text),
        }
        # Persist to vector store and update session state
        fetch_gdelt._store_chunks(result)
        STATE.set_article(result)
        STATE.recent_chunks = fetch_gdelt._chunk_text(text)
        STATE.last_query = topic
        return (
            f"Loaded: {result['title']} — {url}. Say \"read it\" or \"summarize it\"."
        )

    STATE.clear_article()
    if articles:
        return "Couldn't fetch a readable article; try another topic."
    return f"No articles found for '{topic}'."


def handle_read() -> str:
    """Return up to 2000 characters from the last fetched article."""

    article = STATE.last_article
    if not article or not article.get("text"):
        return "No article loaded. Fetch one with handle_fetch()."
    return article["text"][:2000]


def handle_summarize() -> str:
    """Return or generate a summary of the last fetched article."""

    article = STATE.last_article
    if not article:
        return "No article loaded. Fetch one with handle_fetch()."
    summary = article.get("summary")
    if summary:
        return summary
    text = article.get("text")
    if not text:
        return "No article text to summarize."
    fetch_gdelt = _get_fetch_gdelt()
    summary = fetch_gdelt.summarize(text)
    article["summary"] = summary
    return summary


def handle_memory_search(query: str) -> str:
    """Search stored article chunks for ``query`` with graceful fallback."""

    results = vector_store.query(query)
    if results:
        lines: List[str] = []
        for match in results:
            meta = match.get("metadata", {})
            title = meta.get("title", "")
            url = meta.get("url", "")
            lines.append(f"{title} — {url}".strip())
        return "\n".join(lines)

    if not vector_store.available():
        logger.warning(
            "Vector DB unavailable or embedding model missing; using session memory only."
        )
        matches = [
            chunk for chunk in STATE.recent_chunks if query.lower() in chunk.lower()
        ]
        if matches:
            return "\n".join(matches[:5])

    return "No matches found."


__all__ = [
    "handle_fetch",
    "handle_read",
    "handle_summarize",
    "handle_memory_search",
]
