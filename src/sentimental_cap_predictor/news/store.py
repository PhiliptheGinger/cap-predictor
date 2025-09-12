from __future__ import annotations

"""Lightweight SQLite persistence for fetched articles.

This module defines three tables:

``articles``
    Stores basic article metadata such as title and URL.
``contents``
    Holds extracted article text and optional summaries.
``errors``
    Records failures during fetching or extraction for traceability.

Only the ``url`` field is required for articles and contents, making it a
convenient natural primary key.  The schema is created on first use and the
upsert helpers rely on SQLite's ``ON CONFLICT`` clause so repeated runs simply
update existing records.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Any

DB_PATH = Path(os.getenv("NEWS_DB_PATH", "news.sqlite"))


def _connect() -> sqlite3.Connection:
    """Return a connection to the news database, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # enable foreign key enforcement
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            title TEXT,
            domain TEXT,
            language TEXT,
            seendate TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contents (
            url TEXT PRIMARY KEY REFERENCES articles(url),
            text TEXT,
            summary TEXT,
            sentiment REAL,
            relevance REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS errors (
            url TEXT,
            stage TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def upsert_article(data: Dict[str, Any]) -> None:
    """Insert or update an article record.

    Parameters
    ----------
    data:
        Mapping containing at least ``url``.  Optional keys include ``title``,
        ``domain``, ``language`` and ``seendate``.
    """
    if not data.get("url"):
        return
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO articles (url, title, domain, language, seendate)
            VALUES (:url, :title, :domain, :language, :seendate)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                domain=excluded.domain,
                language=excluded.language,
                seendate=excluded.seendate
            """,
            data,
        )


def upsert_content(
    url: str,
    text: str,
    summary: str | None = None,
    sentiment: float | None = None,
    relevance: float | None = None,
) -> None:
    """Insert or update article content for ``url``."""

    if not url:
        return
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO contents (url, text, summary, sentiment, relevance)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                text=excluded.text,
                summary=excluded.summary,
                sentiment=excluded.sentiment,
                relevance=excluded.relevance
            """,
            (url, text, summary or "", sentiment, relevance),
        )


def log_error(url: str, stage: str, message: str) -> None:
    """Record a fetch or extraction error."""
    if not url:
        url = ""  # errors may occur before a URL is known
    with _connect() as conn:
        conn.execute(
            "INSERT INTO errors (url, stage, message) VALUES (?, ?, ?)",
            (url, stage, message),
        )


__all__ = ["upsert_article", "upsert_content", "log_error", "DB_PATH"]
