from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Protocol

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
    query: str,
    start_date: str,
    end_date: str,
    *,
    max_records: int = 100,
    language: str | None = None,
) -> pd.DataFrame:
    """Query the GDELT API for articles matching ``query`` within a window."""

    url = os.getenv(
        "GDELT_API_URL",
        "https://api.gdeltproject.org/api/v2/doc/doc",
    )
    params = {
        "query": f"{query} lang:{language}" if language else query,
        "mode": "artlist",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": max_records,
        "format": "json",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    try:
        data = response.json()
    except (
        requests.exceptions.JSONDecodeError,
        ValueError,
    ) as exc:  # pragma: no cover - depends on requests
        text = getattr(response, "text", "")
        msg = f"Failed to parse GDELT response as JSON: {exc}"
        if text:
            msg += f" Response text: {text}"
        raise RuntimeError(msg) from exc
    articles = data.get("articles", [])
    return pd.DataFrame(articles)


def extract_article_content(
    url: str,
    use_headless: bool = False,
) -> Optional[str]:
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
        agent = headless_agent if use_headless else default_agent
        config.browser_user_agent = agent
        config.request_timeout = 10
        article = Article(url, config=config, keep_article_html=True)
        article.download()
        article.parse()
        return article.text
    except (ArticleException, requests.exceptions.RequestException) as exc:
        logger.error("Error extracting content from %s: %s", url, exc)
        return None
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.exception(
            "Unexpected error extracting content from %s: %s",
            url,
            exc,
        )
        raise


@dataclass
class FetchArticleSpec:
    """Specification for fetching and validating an article.

    The ``language`` field filters articles both in the initial GDELT query
    and again during candidate validation, providing a safeguard against
    mislabeled results that slip through the API filter.
    """

    query: str
    days: int = 1
    max_records: int = 100
    must_contain_any: tuple[str, ...] = ()
    # Some domains require authentication which prevents article extraction.
    # ``seekingalpha.com`` is included by default as its content is typically
    # behind a paywall.  Additional domains may be supplied by callers.
    avoid_domains: tuple[str, ...] = ("seekingalpha.com",)
    require_text_accessible: bool = False
    novelty_against_urls: tuple[str, ...] = ()
    language: str | None = "english"


def is_valid_candidate(article: pd.Series, spec: FetchArticleSpec) -> bool:
    """Return ``True`` if ``article`` satisfies keyword, domain and language
    filters."""

    title = str(article.get("title") or article.get("headline") or "")
    url = str(article.get("url") or "")
    article_language = str(article.get("language") or "").lower()
    spec_language = str(spec.language or "").lower()

    # Treat ISO codes and full language names as equivalent by comparing the
    # first two characters of each.  This allows values like ``"en"`` and
    # ``"english"`` to match while still rejecting mismatched languages.
    article_lang_code = article_language[:2]
    spec_lang_code = spec_language[:2]
    language_mismatch = (
        spec_language and article_lang_code and (article_lang_code != spec_lang_code)
    )
    if language_mismatch:
        return False
    if spec.must_contain_any:
        lower = title.lower()
        if not any(k.lower() in lower for k in spec.must_contain_any):
            return False
    if spec.avoid_domains and url:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.lower()
        if any(domain.endswith(d.lower()) for d in spec.avoid_domains):
            return False
    return True


def is_probably_login_page(text: str) -> bool:
    """Heuristic check for login/paywall pages.

    Some sites return HTML that contains a sign-in prompt instead of the
    article body.  These pages typically include phrases like ``sign in`` or
    ``subscribe``.  When such phrases are detected the caller should treat the
    content as unreadable.
    """

    lower = text.lower()
    login_phrases = ("sign in", "subscribe", "enable javascript")
    return any(p in lower for p in login_phrases)


def try_to_extract(url: str) -> Optional[str]:
    """Attempt to extract article text.

    The helper performs multiple extraction attempts using different user
    agents.  ``None`` is returned when no readable text could be obtained.
    """

    if not url:
        return None

    # First attempt with the default user agent
    text = extract_article_content(url)
    if text and text.strip() and not is_probably_login_page(text):
        return text

    # Second attempt using a headless browser user agent which can sometimes
    # bypass simple bot protections.
    text = extract_article_content(url, use_headless=True)
    if text and text.strip() and not is_probably_login_page(text):
        return text

    return None


def novelty_score(article: dict, spec: FetchArticleSpec) -> float:
    """Simple score indicating how novel ``article`` is."""

    if not spec.novelty_against_urls:
        return 1.0
    url = article.get("url", "")
    return 0.0 if url in spec.novelty_against_urls else 1.0


def title_novelty(title: str, seen_titles: Iterable[str]) -> float:
    """Return a novelty score based on similarity to ``seen_titles``."""

    if not title or not seen_titles:
        return 1.0
    from difflib import SequenceMatcher

    max_sim = max(SequenceMatcher(None, title, t).ratio() for t in seen_titles)
    return 1.0 - max_sim


def rank_candidates(
    candidates: list[dict],
    spec: FetchArticleSpec,
    seen_titles: Iterable[str] = (),
) -> list[dict]:
    """Rank candidates prioritising accessible text and novelty."""

    return sorted(
        candidates,
        key=lambda c: (
            bool(c.get("content")),
            novelty_score(c, spec),
            title_novelty(c.get("title", ""), seen_titles),
        ),
        reverse=True,
    )


def fetch_article(
    spec: FetchArticleSpec, *, seen_titles: Iterable[str] = ()
) -> ArticleData:
    """Fetch an article using ``spec`` applying validations and ranking."""

    end = datetime.utcnow()
    start = end - timedelta(days=spec.days)
    df = query_gdelt_for_news(
        spec.query,
        start.strftime("%Y%m%d%H%M%S"),
        end.strftime("%Y%m%d%H%M%S"),
        max_records=spec.max_records,
        language=spec.language,
    )
    if df.empty:
        raise RuntimeError("No articles returned for query")

    candidates: list[dict] = []
    for _, row in df.iterrows():
        if not is_valid_candidate(row, spec):
            continue
        url = row.get("url") or ""
        text = try_to_extract(url)
        if spec.require_text_accessible and not text:
            continue
        candidates.append(
            {
                "title": row.get("title") or row.get("headline") or "",
                "url": url,
                "content": text,
            }
        )

    if not candidates:
        raise RuntimeError("No candidate articles matched filters")

    for candidate in rank_candidates(candidates, spec, seen_titles):
        if candidate.get("content"):
            return ArticleData(
                title=candidate["title"],
                url=candidate["url"],
                content=candidate["content"] or "",
            )
    raise RuntimeError("No readable article found")


def fetch_first_gdelt_article(
    query: str,
    *,
    prefer_content: bool = True,
    days: int = 1,
    max_records: int = 100,
) -> ArticleData:
    """Backward compatible wrapper around :func:`fetch_article`."""

    spec = FetchArticleSpec(
        query=query,
        days=days,
        max_records=max_records,
        require_text_accessible=prefer_content,
    )
    return fetch_article(spec)


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
    "FetchArticleSpec",
    "query_gdelt_for_news",
    "extract_article_content",
    "fetch_article",
    "fetch_first_gdelt_article",
    "fetch_news",
    "fetch_headline",
]
