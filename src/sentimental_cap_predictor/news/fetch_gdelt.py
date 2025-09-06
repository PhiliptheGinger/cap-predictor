from __future__ import annotations

import argparse
import json
import textwrap

import requests

from .extract import extract_main_text, fetch_html

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


def search_gdelt(query: str, max_records: int = 10):
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


def summarize(text: str, max_chars: int = 800) -> str:
    text = " ".join(text.split())
    return textwrap.shorten(text, width=max_chars, placeholder="…")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--max", type=int, default=10)
    args = ap.parse_args()

    arts = search_gdelt(args.query, args.max)
    for art in arts:
        url = art.get("url")
        if not url:
            continue

        # prefer friendlier domains if possible
        if not domain_ok(url):
            continue

        try:
            html = fetch_html(url)
            text = extract_main_text(html, url=url)
            if not text:
                # couldn’t extract; skip to next
                continue

            result = {
                "title": art.get("title", ""),
                "url": url,
                "sourceDomain": art.get("domain", ""),
                "language": art.get("language", ""),
                "seendate": art.get("seendate", ""),
                "text": text,
                "summary": summarize(text),
            }
            print(json.dumps(result, ensure_ascii=False))
            return
        except requests.HTTPError:
            # 403/404/etc. Skip and try next record.
            continue
        except Exception:
            continue

    # If nothing worked, at least return an empty envelope
    print(json.dumps({"text": "", "summary": ""}))


if __name__ == "__main__":
    main()
