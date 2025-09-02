"""Connector for fetching articles from the PubMed API.

This module provides helper functions for querying the PubMed
E-utilities API and storing a subset of each article's metadata as
JSON.  Only the ``id`` (PMID), ``title`` and ``abstract`` fields are
retained to keep the payload small and portable.

The implementation deliberately avoids external dependencies beyond
``requests`` and ``loguru`` and is resilient to temporary network
issues.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _parse_articles(xml_text: str) -> List[Dict[str, str]]:
    """Parse a PubMed XML document into a list of articles."""

    root = ET.fromstring(xml_text)
    articles: List[Dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="").strip()
        abstract_parts: List[str] = []
        for elem in article.findall(".//Abstract/AbstractText"):
            abstract_parts.append("".join(elem.itertext()).strip())
        abstract = " ".join(abstract_parts)
        articles.append({"id": pmid, "title": title, "abstract": abstract})
    return articles


def fetch(
    query: str = "cancer",
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """Fetch PubMed articles matching *query*.

    Parameters
    ----------
    query:
        PubMed search query.  Defaults to ``"cancer"``.
    max_results:
        Maximum number of results to return.  Defaults to ``10``.
    """

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    logger.debug("Querying PubMed: %s", params)
    search_resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    search_resp.raise_for_status()
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    }
    logger.debug("Fetching PubMed details: %s", fetch_params)
    fetch_resp = requests.get(EFETCH_URL, params=fetch_params, timeout=30)
    fetch_resp.raise_for_status()
    return _parse_articles(fetch_resp.text)


def update_store(
    path: Path,
    query: str = "cancer",
    max_results: int = 10,
) -> Path:
    """Fetch articles and persist them to *path* as JSON."""

    articles = fetch(query=query, max_results=max_results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(articles, indent=2))
    logger.info("Saved %d PubMed articles to %s", len(articles), path)
    return path


__all__ = ["fetch", "update_store"]
