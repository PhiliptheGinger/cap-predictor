"""Client helpers for querying the GDELT news API."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests

from sentimental_cap_predictor.config import GDELT_API_URL


logger = logging.getLogger(__name__)

DISALLOWED_DOMAINS = ("wsj.com", "ft.com", "bloomberg.com")


def search_gdelt(
    query: str, max_results: int = 3, *, raise_errors: bool = False
) -> list[dict]:
    """Search the GDELT ``doc`` endpoint for ``query``.

    Parameters
    ----------
    query:
        Search string passed to GDELT.
    max_results:
        Maximum number of articles to return.  The value is also forwarded to
        the API via the ``maxrecords`` parameter.

    Returns
    -------
    list[dict]
        A list of dictionaries with keys ``title``, ``url``, ``source`` and
        ``pubdate``.  An empty list is returned when the request fails or the
        response payload cannot be parsed.
    """

    params = {
        "query": query,
        "format": "json",
        "mode": "artlist",
        "maxrecords": max_results,
    }
    try:
        response = requests.get(GDELT_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as err:
        logger.warning("GDELT request failed: %s", err)
        if raise_errors:
            raise
        return []

    try:
        data = response.json()
    except ValueError as err:
        logger.warning("GDELT response JSON decode failed: %s", err)
        if raise_errors:
            raise
        return []

    articles = data.get("articles", [])
    results: list[dict] = []
    for article in articles:
        url = article.get("url", "")
        domain = urlparse(url).netloc
        if any(d in domain for d in DISALLOWED_DOMAINS):
            logger.info("Skipping %s: disallowed domain", domain)
            continue
        results.append(
            {
                "title": article.get("title", ""),
                "url": url,
                "source": (
                    article.get("sourceCommonName")
                    or article.get(
                        "source",
                        "",
                    )
                ),
                "pubdate": (
                    article.get("seendate") or article.get("publishedDate", "")
                ),
            }
        )
        if len(results) >= max_results:
            break
    return results
