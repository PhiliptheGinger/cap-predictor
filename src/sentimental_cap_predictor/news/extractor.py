"""Robust article text extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import trafilatura
from readability import Document
from trafilatura.settings import use_config

CONFIG = use_config()
CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


@dataclass
class ExtractedArticle:
    text: str
    title: str | None
    byline: str | None
    date: str | None


class ArticleExtractor:
    """Extract main article content from HTML."""

    def extract(self, html: str, url: str | None = None) -> ExtractedArticle | None:
        data = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            config=CONFIG,
            output="json",
        )
        if data:
            import json

            meta = json.loads(data)
            return ExtractedArticle(
                meta.get("text", "").strip(),
                meta.get("title"),
                meta.get("author"),
                meta.get("date"),
            )

        try:
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            text = trafilatura.extract(summary_html) or ""
            return ExtractedArticle(text.strip(), doc.title(), None, None)
        except Exception:
            return None


__all__ = ["ArticleExtractor", "ExtractedArticle"]
