"""Connector for the OpenAlex API.

Only a handful of fields are persisted for each work: ``id``, ``title`` and
``abstract``.  The latter is reconstructed from the ``abstract_inverted_index``
field returned by the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

OPENALEX_URL = "https://api.openalex.org/works"


def _abstract_from_inverted_index(
    inv_index: Dict[str, List[int]] | None,
) -> str:
    if not inv_index:
        return ""
    pairs: List[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            pairs.append((pos, word))
    return " ".join(word for pos, word in sorted(pairs))


def fetch(
    search: str = "machine learning",
    per_page: int = 25,
) -> List[Dict[str, str]]:
    """Fetch works from OpenAlex matching *search*.

    Parameters
    ----------
    search:
        The search term to use.  Defaults to ``"machine learning"``.
    per_page:
        Number of results to retrieve.  Defaults to ``25``.
    """

    params = {"search": search, "per-page": per_page}
    logger.debug("Querying OpenAlex: %s", params)
    response = requests.get(OPENALEX_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    works = []
    for work in data.get("results", []):
        works.append(
            {
                "id": work.get("id"),
                "title": work.get("display_name", ""),
                "abstract": _abstract_from_inverted_index(
                    work.get("abstract_inverted_index")
                ),
            }
        )
    return works


def update_store(
    path: Path,
    search: str = "machine learning",
    per_page: int = 25,
) -> Path:
    """Fetch works and persist them to *path* as JSON."""

    works = fetch(search=search, per_page=per_page)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(works, indent=2))
    logger.info("Saved %d OpenAlex works to %s", len(works), path)
    return path


__all__ = ["fetch", "update_store"]
