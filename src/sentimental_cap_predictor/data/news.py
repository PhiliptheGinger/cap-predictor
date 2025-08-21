from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


class NewsSource(Protocol):
    """Protocol for news data sources."""

    def fetch(self, ticker: str) -> pd.DataFrame:
        ...


@dataclass
class FileSource:
    """Read news data from a local CSV file.

    The CSV is expected to contain columns ``date``, ``headline``, ``source`` and
    optionally ``ticker``. Rows are filtered by ``ticker`` when that column is
    present.
    """

    path: Path = Path("data/news.csv")

    def fetch(self, ticker: str) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["date", "headline", "source"])
        df = pd.read_csv(self.path, parse_dates=["date"])
        if "ticker" in df.columns:
            df = df[df["ticker"] == ticker]
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
