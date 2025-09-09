from __future__ import annotations

import argparse
import json
import logging
import textwrap
from urllib.parse import urlparse

import requests

from .extract import extract_main_text, fetch_html

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/118.0.0.0 Safari/537.36"
)

# Sites that are generally scraper-friendly
PREFERRED_DOMAINS = (
    "apnews.com",
    "reuters.com",
    "bbc.com",
    "cnn.com",
    "nytimes.com",  # still mixed; keep but expect paywalls/403
    "theguardian.com",
    "npr.org",
    "abcnews.go.com",
    "cbc.ca",
)

# Sites that are known to block scraping or are paywalled
BLOCKED_DOMAINS = (
    "wsj.com",
    "bloomberg.com",
    "ft.com",
)


def search_gdelt(query: str, max_records: int = 15):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    r = requests.get(
        GDELT_DOC_API, params=params, headers={"User-Agent": UA}, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    return data.get("articles", [])


def domain_ok(url: str) -> bool:
    return any(d in url for d in PREFERRED_DOMAINS)


def domain_blocked(url: str) -> bool:
    return any(d in url for d in BLOCKED_DOMAINS)


def summarize(text: str, max_chars: int = 800) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(text, width=max_chars, placeholder="…")


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--max", type=int, default=15)
    args = ap.parse_args()

    arts = search_gdelt(args.query, args.max)
    for art in arts:
        url = art.get("url")
        if not url:
            logger.info("Skipping record with no URL")
            continue

        domain = urlparse(url).netloc
        if domain_blocked(url):
            logger.info("Skipping %s: blocked domain", domain)
            continue
        if not domain_ok(url):
            logger.info("Skipping %s: undesired domain", domain)
            continue

        try:
            html = fetch_html(url)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            logger.info("HTTP %s for %s", status, domain)
            continue
        except Exception as e:
            logger.info("Error fetching %s: %s", domain, e)
            continue

        text = extract_main_text(html, url=url)
        if not text:
            logger.info("Skipping %s: no extractable text", domain)
            continue

        result = {
            "title": art.get("title", ""),
            "url": url,
            "domain": art.get("domain", ""),
            "language": art.get("language", ""),
            "seendate": art.get("seendate", ""),
            "text": text,
            "summary": summarize(text),
        }
        print(json.dumps(result, ensure_ascii=False))
        return

    # If nothing worked, at least return an empty envelope
    print(json.dumps({}))


if __name__ == "__main__":
    main()
