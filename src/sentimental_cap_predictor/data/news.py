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

    def fetch(self, ticker: str) -> pd.DataFrame: ...


@dataclass
class FileSource:
    """Read news data from a local CSV file.

    The CSV is expected to contain columns ``date``, ``headline`` and
    ``source`` plus an optional ``ticker`` column. Rows are filtered by
    ``ticker`` when that column is present.
    """

    path: Path = Path("data/news.csv")

    def fetch(self, ticker: str) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["date", "headline", "source"])
        df = pd.read_csv(self.path, parse_dates=["date"])
        if "ticker" in df.columns:
            df = df[df["ticker"] == ticker]
        return df[["date", "headline", "source"]]


@dataclass
class GDELTSource:
    """Fetch news headlines from the GDELT API.

    The source queries GDELT's `doc` endpoint for articles mentioning the
    provided ``ticker`` over the most recent ``days`` window. Only a subset of
    fields is returned to keep the resulting dataframe lightweight.
    """

    days: int = 7
    max_records: int = 100
    api_url: str = os.getenv(
        "GDELT_API_URL", "https://api.gdeltproject.org/api/v2/doc/doc"
    )

    def fetch(self, ticker: str) -> pd.DataFrame:
        end = datetime.utcnow()
        start = end - timedelta(days=self.days)
        params = {
            "query": ticker,
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


def fetch_news(ticker: str, source: NewsSource | None = None) -> pd.DataFrame:
    """Fetch news headlines for ``ticker``.

    Parameters
    ----------
    ticker:
        Instrument symbol to fetch headlines for.
    source:
        Optional data source implementing :class:`NewsSource`. Defaults to
        :class:`FileSource` reading ``data/news.csv``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``date``, ``headline`` and ``source``.
    """

    source = source or FileSource()
    df = source.fetch(ticker)
    return df[["date", "headline", "source"]]


__all__ = ["NewsSource", "FileSource", "GDELTSource", "fetch_news"]
