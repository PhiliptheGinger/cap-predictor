from __future__ import annotations

import requests

try:
    from readability import Document  # optional fallback
except Exception:
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


def fetch_html(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_main_text(html: str, url: str | None = None) -> str:
    # 1) Try trafilatura first
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        config=CONFIG,
    )
    if text:
        return text.strip()

    # 2) Fallback to readability if available
    if Document is not None:
        try:
            doc = Document(html)
            # readability gives you a cleaned HTML;
            # strip tags quickly via trafilatura
            cleaned = trafilatura.extract(
                doc.summary(html_partial=True),
                url=url,
            )
            if cleaned:
                return cleaned.strip()
            return doc.summary().strip()
        except Exception:
            pass

    # 3) If all else fails…
    return ""
