from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd
import requests
from newspaper import Article, ArticleException, Config

logger = logging.getLogger(__name__)


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


@dataclass
class ArticleData:
    """Container for article information returned by GDELT."""

    title: str = ""
    url: str = ""
    content: str = ""


def query_gdelt_for_news(
    query: str, start_date: str, end_date: str, *, max_records: int = 100
) -> pd.DataFrame:
    """Query the GDELT API for articles matching ``query`` within a window."""

    url = os.getenv("GDELT_API_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
    params = {
        "query": query,
        "mode": "artlist",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": max_records,
        "format": "json",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    articles = data.get("articles", [])
    return pd.DataFrame(articles)


def extract_article_content(url: str, use_headless: bool = False) -> Optional[str]:
    """Extract main content from a news article using newspaper3k."""

    try:
        config = Config()
        headless_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) HeadlessChrome/120.0.0 Safari/537.36"
        )
        default_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        )
        config.browser_user_agent = headless_agent if use_headless else default_agent
        config.request_timeout = 10
        article = Article(url, config=config, keep_article_html=True)
        article.download()
        article.parse()
        return article.text
    except (ArticleException, requests.exceptions.RequestException) as exc:
        logger.error("Error extracting content from %s: %s", url, exc)
        return None
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception("Unexpected error extracting content from %s: %s", url, exc)
        raise


def fetch_first_gdelt_article(
    query: str,
    *,
    prefer_content: bool = True,
    days: int = 1,
    max_records: int = 100,
) -> ArticleData:
    """Return information on the first GDELT article matching ``query``."""

    end = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%d%H%M%S")
    df = query_gdelt_for_news(query, start, end, max_records=max_records)
    if df.empty:
        return ArticleData()

    article = df.iloc[0]
    title = article.get("title") or article.get("headline") or ""
    url = article.get("url") or ""
    content = ""
    if prefer_content and url:
        text = extract_article_content(url)
        if text:
            content = text
    return ArticleData(title=title, url=url, content=content)


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


__all__ = [
    "NewsSource",
    "FileSource",
    "GDELTSource",
    "ArticleData",
    "query_gdelt_for_news",
    "extract_article_content",
    "fetch_first_gdelt_article",
    "fetch_news",
    "fetch_headline",
]
