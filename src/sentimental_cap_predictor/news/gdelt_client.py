"""Client helpers for querying the GDELT news API with proxy control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from urllib.parse import urlparse

import httpx

from sentimental_cap_predictor.config import GDELT_API_URL


logger = logging.getLogger(__name__)


@dataclass
class ArticleStub:
    """Minimal information about an article returned by GDELT."""

    url: str
    title: str | None
    domain: str
    seendate: datetime | None
    source: str | None = None


class GdeltClient:
    """Simple client for the GDELT 2.0 DOC API.

    Parameters
    ----------
    timeout_s:
        Request timeout in seconds.
    use_env_proxy:
        If ``True`` respect ``HTTP(S)_PROXY`` environment variables.  When
        ``False`` (the default) proxies from the environment are ignored which
        avoids the ``ProxyError('Cannot connect to proxy')`` failures seen in
        restricted environments.
    """

    def __init__(self, timeout_s: float = 10.0, use_env_proxy: bool = False) -> None:
        self.client = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": "cap-predictor/1.0"},
            trust_env=use_env_proxy,
        )

    def search(
        self, query: str, *, timespan: str = "24H", max_records: int = 75
    ) -> list[ArticleStub]:
        """Return a list of :class:`ArticleStub` for ``query``."""

        params = {
            "query": query,
            "timespan": timespan,
            "maxrecords": max_records,
            "format": "json",
            "mode": "artlist",
        }
        resp = self.client.get(GDELT_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[ArticleStub] = []
        for art in data.get("articles", []):
            url = art.get("url")
            if not url:
                continue
            domain = art.get("domain") or urlparse(url).netloc
            raw_seen = art.get("seendate")
            seendate = None
            if raw_seen:
                try:
                    seendate = datetime.fromisoformat(raw_seen.replace("Z", "+00:00"))
                except ValueError:
                    seendate = None
            stub = ArticleStub(
                url=url,
                title=art.get("title"),
                domain=domain,
                seendate=seendate,
                source=art.get("sourceCommonName") or art.get("source"),
            )
            results.append(stub)
            if len(results) >= max_records:
                break
        return results


DISALLOWED_DOMAINS = ("wsj.com", "ft.com", "bloomberg.com")


def search_gdelt(
    query: str, max_results: int = 3, *, raise_errors: bool = False
) -> list[dict]:
    """Compatibility wrapper returning dictionaries instead of dataclasses."""

    client = GdeltClient()
    try:
        stubs = client.search(query, max_records=max_results)
    except httpx.HTTPError as err:
        logger.warning("GDELT request failed: %s", err)
        if raise_errors:
            raise
        return []

    results: list[dict] = []
    for stub in stubs:
        if any(d in stub.domain for d in DISALLOWED_DOMAINS):
            logger.info("Skipping %s: disallowed domain", stub.domain)
            continue
        results.append(
            {
                "title": stub.title or "",
                "url": stub.url,
                "source": stub.source or stub.domain,
                "pubdate": stub.seendate.isoformat() if stub.seendate else "",
            }
        )
    return results


__all__ = ["ArticleStub", "GdeltClient", "search_gdelt"]
