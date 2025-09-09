from __future__ import annotations

import logging

import requests

try:
    from readability import Document  # optional fallback
except Exception:  # pragma: no cover - optional dependency
    Document = None

import trafilatura
from trafilatura.settings import use_config

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/118.0.0.0 Safari/537.36"
)

CONFIG = use_config()
# make extraction non-finicky about timeouts
CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")


logger = logging.getLogger(__name__)


def fetch_html(url: str, timeout: int = 20) -> str:
    """Fetch raw HTML from ``url`` using a desktop User-Agent."""

    resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)

    if resp.status_code in {403, 404}:
        logger.info("%s returned status %s", url, resp.status_code)
    else:
        logger.info("Fetched %s with status %s", url, resp.status_code)

    resp.raise_for_status()
    return resp.text


def extract_main_text(html: str, url: str | None = None) -> str:
    """Extract the main textual content from HTML.

    Trafilatura is attempted first; if it fails, a readability-lxml fallback
    is used and the cleaned HTML is passed through trafilatura again. When no
    text can be recovered an empty string is returned.
    """

    # 1) Try trafilatura first
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        config=CONFIG,
    )
    if text:
        logger.info("Extracted text via trafilatura: %s", url or "<html>")
        return text.strip()

    # 2) Fallback to readability if available
    if Document is not None:
        try:
            doc = Document(html)
            cleaned_html = doc.summary(html_partial=True)
            cleaned = trafilatura.extract(cleaned_html, url=url, config=CONFIG)
            if cleaned:
                logger.info(
                    "Extracted main text for %s via readability fallback",
                    url or "<html>",
                )
                return cleaned.strip()
        except Exception:  # pragma: no cover - extraction is best effort
            pass

    logger.info("Failed to extract main text for %s", url or "<html>")
    return ""
