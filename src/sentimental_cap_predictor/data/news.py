from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests


class NewsSource(Protocol):
    """Protocol for news data sources."""

    def fetch(self, query: str) -> pd.DataFrame: ...


@dataclass
class FileSource:
    """Read news data from a local CSV file.

    The CSV is expected to contain columns ``date``, ``headline`` and
    ``source`` plus an optional ``query`` column. For backwards
    compatibility, a column named ``ticker`` is also recognised. Rows are
    filtered by the provided ``query`` when that column is present.
    """

    path: Path = Path("data/news.csv")

    def fetch(self, query: str) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["date", "headline", "source"])
        df = pd.read_csv(self.path, parse_dates=["date"])
        if "query" in df.columns:
            df = df[df["query"] == query]
        elif "ticker" in df.columns:
            df = df[df["ticker"] == query]
        return df[["date", "headline", "source"]]


@dataclass
class GDELTSource:
    """Fetch news headlines from the GDELT API.

    The source queries GDELT's ``doc`` endpoint for articles matching the
    provided ``query`` over the most recent ``days`` window. Only a subset of
    fields is returned to keep the resulting dataframe lightweight.
    """

    days: int = 7
    max_records: int = 100
    api_url: str = os.getenv(
        "GDELT_API_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
    )

    def fetch(self, query: str) -> pd.DataFrame:
        end = datetime.utcnow()
        start = end - timedelta(days=self.days)
        params = {
            "query": query,
            "mode": "artlist",
            "startdatetime": start.strftime("%Y%m%d000000"),
            "enddatetime": end.strftime("%Y%m%d000000"),
            "maxrecords": self.max_records,
            "format": "json",
        }
        response = requests.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        rows = []
        for article in articles:
            rows.append(
                {
                    "date": pd.to_datetime(article.get("seendate")),
                    "headline": article.get("title", ""),
                    "source": article.get("sourceCommonName")
                    or article.get("source", ""),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["date", "headline", "source"])
        df = pd.DataFrame(rows)
        return df[["date", "headline", "source"]]


def fetch_news(
    query: str | None = None,
    source: NewsSource | None = None,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Fetch news headlines for a search ``query``.

    Parameters
    ----------
    query:
        Search term to fetch headlines for.
    source:
        Optional data source implementing :class:`NewsSource`. Defaults to
        :class:`FileSource` reading ``data/news.csv``.
    ticker:
        Deprecated alias for ``query`` for backwards compatibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``date``, ``headline`` and ``source``.
    """

    if query is None:
        if ticker is None:
            raise TypeError("fetch_news() missing required argument 'query'")
        query = ticker

    source = source or FileSource()
    df = source.fetch(query)
    return df[["date", "headline", "source"]]


def fetch_headline(query: str, source: NewsSource | None = None) -> str:
    """Return the first headline for ``query`` using the provided ``source``.

    When no ``source`` is supplied, :class:`GDELTSource` is used to fetch
    headlines from the GDELT API. An empty string is returned when no results
    are available.
    """

    source = source or GDELTSource(max_records=1)
    df = source.fetch(query)
    if df.empty:
        return ""
    return str(df.iloc[0]["headline"])


__all__ = ["NewsSource", "FileSource", "GDELTSource", "fetch_news", "fetch_headline"]
