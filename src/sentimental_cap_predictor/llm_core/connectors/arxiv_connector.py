"""Connector for downloading papers from the arXiv API.

The module provides small convenience functions for fetching papers
from the public arXiv API and persisting them to a local JSON file.
The payload intentionally stores only a subset of the response
(`id`, `title`, `summary` and `updated`).

These helpers are intentionally lightweight; they avoid introducing
additional dependencies and are tolerant to the API being unavailable.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _parse_feed(xml_text: str) -> List[Dict[str, str]]:
    """Parse an Atom XML feed into a list of dictionaries."""
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        entries.append(
            {
                "id": entry.findtext("atom:id", default="", namespaces=ns),
                "title": entry.findtext(
                    "atom:title", default="", namespaces=ns
                ).strip(),
                "summary": entry.findtext(
                    "atom:summary", default="", namespaces=ns
                ).strip(),
                "updated": entry.findtext(
                    "atom:updated",
                    default="",
                    namespaces=ns,
                ),
            }
        )
    return entries


def fetch(
    query: str = "cat:cs.AI",
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """Fetch papers from arXiv matching *query*.

    Parameters
    ----------
    query:
        arXiv search query.  Defaults to ``"cat:cs.AI"``.
    max_results:
        Maximum number of results to return.  Defaults to ``10``.
    """

    params = {"search_query": query, "start": 0, "max_results": max_results}
    logger.debug("Querying arXiv: %s", params)
    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()
    return _parse_feed(response.text)


def update_store(
    path: Path,
    query: str = "cat:cs.AI",
    max_results: int = 10,
) -> Path:
    """Fetch papers and persist them to *path* as JSON.

    The parent directory of *path* is created automatically.  Returns the
    path written to for convenience.
    """

    papers = fetch(query=query, max_results=max_results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(papers, indent=2))
    logger.info("Saved %d arXiv papers to %s", len(papers), path)
    return path


__all__ = ["fetch", "update_store"]
