"""Utilities for fetching and processing news articles.

This module provides helper functions for downloading web pages, extracting
main article text, stripping advertisement sections, basic text analysis and
chunking.  Heavy dependencies are optional and gracefully degraded when not
available.

Sentiment analysis relies on a lightweight Transformers model,
``distilbert/distilbert-base-uncased-finetuned-sst-2-english`` by default. The
model can be overridden by setting the ``SENTIMENT_MODEL`` environment
variable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ..config import SENTIMENT_MODEL

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def fetch_html(
    url: str,
    timeout: int = 10,
    retries: int = 3,
    backoff_factor: float = 0.5,
    user_agent: str | None = None,
) -> str:
    """Retrieve raw HTML from ``url`` with retry and timeout handling.

    Parameters
    ----------
    url:
        Web address to fetch.
    timeout:
        Timeout in seconds for the request.  Defaults to ``10``.
    retries:
        Number of retry attempts for transient failures.  Defaults to ``3``.
    backoff_factor:
        Factor used for exponential backoff between retry attempts.
        Defaults to ``0.5``.
    user_agent:
        Optional ``User-Agent`` header to send with the request. If not
        provided a generic user agent is used.

    Returns
    -------
    str
        Raw HTML content.  Empty string on failure.
    """

    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {"User-Agent": user_agent or "sentimental-cap-predictor/1.0"}

    try:
        response = session.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""
    finally:
        session.close()


def extract_main(html: str, url: str | None = None) -> str:
    """Extract the main article text from HTML.

    The function first tries :mod:`trafilatura` which performs excellent
    extraction.  If it is not available or fails, it falls back to using
    :mod:`readability` and :class:`~bs4.BeautifulSoup`.  When all parsers fail
    the function returns the raw text with HTML tags stripped.

    Parameters
    ----------
    html:
        Raw HTML document.
    url:
        Optional URL associated with the HTML.  Some extractors can use this
        for improved results.

    Returns
    -------
    str
        Plain text of the article or a simple tag-stripped version if parsing
        fails.
    """

    # Try trafilatura if installed.
    try:  # pragma: no cover - dependency may be missing
        import trafilatura

        text = trafilatura.extract(html, url=url)
        if text:
            return text
    except Exception:
        pass

    # Fall back to readability + BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        from readability import Document
    except ImportError as exc:  # pragma: no cover - optional deps
        if getattr(exc, "name", "") == "readability":
            logger.info("Install readability-lxml to improve article parsing")
        else:
            logger.warning("Failed to extract main text: %s", exc)
    else:
        try:
            doc = Document(html)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, "html.parser")
            return soup.get_text("\n")
        except Exception as exc:  # pragma: no cover - parser errors
            logger.warning("Failed to extract main text: %s", exc)

    # Final fallback: parse HTML with BeautifulSoup to remove script and style
    # elements.  If BeautifulSoup is not available, revert to a simple regex
    # based stripping of tags.
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        return soup.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)


def strip_ads(text: str) -> str:
    """Remove advertisement lines from ``text``.

    Lines containing only words such as ``Advertisement`` or ``Sponsored
    Content`` are removed completely.
    """

    ad_pattern = re.compile(
        r"^\s*(advertisement|sponsored\s+content)\s*$",
        re.I,
    )
    lines = [line for line in text.splitlines() if not ad_pattern.match(line)]
    return "\n".join(lines)


def analyze(text: str, *, include_sentiment: bool = False) -> Dict[str, Any]:
    """Perform lightweight analysis on ``text``.

    The function attempts language detection and entity extraction. Sentiment
    estimation is executed only when ``include_sentiment`` is ``True`` and uses
    the model specified by :data:`SENTIMENT_MODEL`.
    Optional libraries are used when available and the function gracefully
    degrades when dependencies are missing.
    """

    tokens = text.split()

    # Language detection
    lang = "unknown"
    try:  # pragma: no cover - dependency may be missing
        from langdetect import detect

        lang = detect(text)
    except Exception:
        pass

    # Entity extraction using a simple regex fallback.
    # If spaCy is installed we use it for better results.
    entities: List[Any]
    try:  # pragma: no cover - optional dependency
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:  # model not present
            nlp = spacy.blank("en")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception:
        entities = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)

    analysis: Dict[str, Any] = {
        "lang": lang,
        "entities": entities,
        "word_count": len(tokens),
        "tokens": tokens,
    }

    if include_sentiment:
        sentiment: Dict[str, Any] = {"label": "unknown", "score": 0.0}
        try:  # pragma: no cover - heavy dependency
            from transformers import pipeline

            sa = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
            result = sa(text[:512])[0]
            sentiment = {
                "label": result["label"],
                "score": float(result["score"]),
            }
        except Exception:
            pass
        analysis["sentiment"] = sentiment

    return analysis


def chunk(text: str, max_tokens: int, overlap: int = 0) -> List[str]:
    """Split ``text`` into token-based chunks.

    Parameters
    ----------
    text:
        Input text to split.
    max_tokens:
        Maximum number of tokens per chunk.
    overlap:
        Number of tokens that should overlap between consecutive chunks.

    Returns
    -------
    list[str]
        List of chunks.
    """

    if max_tokens <= 0:
        return []

    tokens = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def summarize(text: str, max_sentences: int = 3) -> str:
    """Return a simple summary consisting of the first ``max_sentences``.

    This is a very naive summarizer used as a fallback when no more
    sophisticated summarization library is available.
    """

    sentence_end = re.compile(r"(?<=[.!?]) +")
    sentences = sentence_end.split(text.strip())
    return " ".join(sentences[:max_sentences])


def translate(text: str, target_lang: str) -> str | None:
    """Attempt to translate ``text`` into ``target_lang``.

    If a translation library is not available or translation fails, ``None`` is
    returned.
    """

    try:  # pragma: no cover - optional dependency
        from googletrans import Translator
    except Exception:
        return None

    try:  # pragma: no cover - translation may fail at runtime
        translator = Translator()
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return None
