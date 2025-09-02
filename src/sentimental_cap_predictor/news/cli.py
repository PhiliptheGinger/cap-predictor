"""Command line utilities for news related tasks."""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

import typer

from .article_reader import analyze as analyze_text
from .article_reader import chunk as chunk_text
from .article_reader import extract_main, fetch_html, strip_ads
from .article_reader import summarize as summarize_text
from .article_reader import translate as translate_text
from .gdelt_client import search_gdelt


class TranslateMode(str, Enum):
    off = "off"
    auto = "auto"
    en = "en"


app = typer.Typer(help="News utilities")


@app.command("fetch-gdelt")
def fetch_gdelt_command(
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="Search query for GDELT.",
    ),
    max_results: int = typer.Option(
        3, "--max", "-m", help="Maximum number of articles to return."
    ),
) -> None:
    """Fetch articles from the GDELT API and print as JSON."""
    articles = search_gdelt(query=query, max_results=max_results)
    typer.echo(json.dumps(articles))


@app.command("read")
def read_command(
    url: str = typer.Option(..., "--url", "-u", help="Article URL to fetch."),
    summarize: bool = typer.Option(
        False, "--summarize", help="Return a short summary."
    ),
    analyze: bool = typer.Option(
        False, "--analyze", help="Return basic text analysis."
    ),
    chunks: Optional[int] = typer.Option(
        None, "--chunks", help="Split text into chunks of this many tokens."
    ),
    overlap: int = typer.Option(
        0, "--overlap", help="Number of overlapping tokens between chunks."
    ),
    translate: TranslateMode = typer.Option(
        TranslateMode.off,
        "--translate",
        help="Translation mode: off, auto or en.",
    ),
) -> None:
    """Read an article and optionally process it."""
    html = fetch_html(url)
    text = strip_ads(extract_main(html, url=url))
    original_text = text

    analysis = None
    if analyze or translate == TranslateMode.auto:
        analysis = analyze_text(text)

    if translate == TranslateMode.en:
        text = translate_text(text, "en")
    elif translate == TranslateMode.auto and analysis:
        if analysis.get("lang") != "en":
            text = translate_text(text, "en")

    should_process = any(
        [
            summarize,
            analyze,
            chunks is not None,
            translate != TranslateMode.off,
        ]
    )
    if should_process:
        result: dict[str, object] = {"text": text}
        if translate != TranslateMode.off and text != original_text:
            result["original_text"] = original_text
        if summarize:
            result["summary"] = summarize_text(text)
        if analyze:
            result["analysis"] = analysis
        if chunks is not None:
            result["chunks"] = chunk_text(text, chunks, overlap)
        typer.echo(json.dumps(result))
    else:
        typer.echo(text)


if __name__ == "__main__":
    app()
