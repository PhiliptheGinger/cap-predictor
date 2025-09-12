"""Connector for ingesting articles from RSS feeds and optional NewsAPI."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"


def _parse_feed(xml_text: str) -> List[Dict[str, str]]:
    """Parse a simple RSS/Atom feed into a list of dictionaries."""
    root = ET.fromstring(xml_text)
    entries: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        entries.append(
            {
                "title": item.findtext("title", default="").strip(),
                "link": item.findtext("link", default="").strip(),
                "summary": item.findtext("description", default="").strip(),
                "published": item.findtext("pubDate", default="").strip(),
            }
        )
    # Atom feeds use ``entry`` elements instead of ``item``
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        link = ""
        link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
        if link_elem is not None:
            link = link_elem.get("href", "")
        entries.append(
            {
                "title": entry.findtext("{http://www.w3.org/2005/Atom}title", default="").strip(),
                "link": link,
                "summary": entry.findtext("{http://www.w3.org/2005/Atom}summary", default="").strip(),
                "published": entry.findtext("{http://www.w3.org/2005/Atom}updated", default="").strip(),
            }
        )
    return entries


def fetch_feeds(feed_urls: List[str]) -> List[Dict[str, str]]:
    """Fetch and parse multiple RSS feeds.

    Parameters
    ----------
    feed_urls:
        List of RSS feed URLs to download and parse.
    """

    articles: List[Dict[str, str]] = []
    for url in feed_urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            articles.extend(_parse_feed(response.text))
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("Failed to fetch %s: %s", url, exc)
    return articles


def load_feed_list(path: Path) -> List[str]:
    """Load feed URLs from a text file (one per line)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def fetch_newsapi(api_key: str, query: str = "") -> List[Dict[str, str]]:
    """Fetch articles using the NewsAPI.org service."""
    params = {"apiKey": api_key}
    if query:
        params["q"] = query
    response = requests.get(NEWSAPI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    articles: List[Dict[str, str]] = []
    for article in data.get("articles", []):
        articles.append(
            {
                "title": article.get("title", ""),
                "link": article.get("url", ""),
                "summary": article.get("description", ""),
                "published": article.get("publishedAt", ""),
            }
        )
    return articles


def fetch_all(feed_urls: List[str], query: str = "") -> List[Dict[str, str]]:
    """Fetch articles from RSS feeds and NewsAPI if a key is available."""
    articles = fetch_feeds(feed_urls)
    api_key = os.getenv("NEWSAPI_API_KEY")
    if api_key:
        try:
            articles.extend(fetch_newsapi(api_key, query=query))
        except Exception as exc:  # pragma: no cover - logging only
            logger.warning("NewsAPI fetch failed: %s", exc)
    return articles


def fetch_from_file(path: Path, query: str = "") -> List[Dict[str, str]]:
    """Convenience wrapper to fetch feeds listed in *path*."""
    return fetch_all(load_feed_list(path), query=query)


__all__ = [
    "fetch_feeds",
    "fetch_newsapi",
    "fetch_all",
    "fetch_from_file",
    "load_feed_list",
]
