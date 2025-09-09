from __future__ import annotations

import argparse
import json
import logging
import textwrap
from urllib.parse import urlparse

import requests

from sentimental_cap_predictor.memory import vector_store
from .extract import extract_main_text, fetch_html

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/118.0.0.0 Safari/537.36"
)

# Sites that are generally scraper-friendly
PREFERRED_DOMAINS = (
    "apnews.com",
    "reuters.com",
    "bbc.com",
    "cnn.com",
    "nytimes.com",  # still mixed; keep but expect paywalls/403
    "theguardian.com",
    "npr.org",
    "abcnews.go.com",
    "cbc.ca",
)

# Sites that are known to block scraping or are paywalled
BLOCKED_DOMAINS = (
    "wsj.com",
    "bloomberg.com",
    "ft.com",
)


_EMPTY_HTML = {
    "<html></html>",
    "<html><head></head><body></body></html>",
    "",
}


def _is_empty_page(html: str) -> bool:
    """Return ``True`` for trivially empty HTML pages."""

    return "".join(html.split()).lower() in _EMPTY_HTML


def search_gdelt(query: str, max_records: int = 15):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    try:
        r = requests.get(
            GDELT_DOC_API, params=params, headers={"User-Agent": UA}, timeout=30
        )
        r.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        logger.warning("GDELT API request failed: %s", exc)
        return []
    data = r.json()
    articles = []
    for art in data.get("articles", []):
        url = art.get("url")
        if not url:
            continue
        if domain_blocked(url):
            logger.info("Skipping %s: blocked domain", urlparse(url).netloc)
            continue
        try:  # pragma: no cover - network failure
            html = fetch_html(url)
        except Exception:
            continue
        if _is_empty_page(html):
            continue
        articles.append(art)
        if len(articles) >= max_records:
            break
    return articles


def domain_ok(url: str) -> bool:
    return any(d in url for d in PREFERRED_DOMAINS)


def domain_blocked(url: str) -> bool:
    return any(d in url for d in BLOCKED_DOMAINS)


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


def main():
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

        domain = urlparse(url).netloc
        if domain_blocked(url):
            logger.info("Skipping %s: blocked domain", domain)
            continue
        if not domain_ok(url):
            logger.info("Skipping %s: undesired domain", domain)
            continue

        try:
            html = fetch_html(url)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            logger.warning("HTTP %s for %s", status, domain)
            continue
        except Exception as e:
            logger.warning("Error fetching %s: %s", domain, e)
            continue
        if _is_empty_page(html):
            logger.info("Skipping %s: empty page", domain)
            continue

        text = extract_main_text(html, url=url)
        if not text:
            logger.info("Skipping %s: no extractable text", domain)
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


if __name__ == "__main__":
    main()
