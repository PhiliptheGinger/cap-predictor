"""Web search tool implementation using DuckDuckGo."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from ..tool_registry import ToolSpec, register_tool

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - import time dependency errors
    DDGS = None  # type: ignore[assignment]


class WebSearchInput(BaseModel):
    """Input payload for the web search tool."""

    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """Single search result returned by :func:`search_web`."""

    title: str
    snippet: str
    url: str


class WebSearchOutput(BaseModel):
    """Output payload containing search results."""

    results: List[SearchResult]


def search_web(query: str, top_k: int = 5) -> List[dict[str, str]]:
    """Search the web using DuckDuckGo.

    Parameters
    ----------
    query:
        Search terms.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list of dict
        Each dict contains ``title``, ``snippet`` and ``url`` keys.
    """

    if DDGS is None:  # pragma: no cover - runtime dependency
        raise RuntimeError("duckduckgo_search is required for web search")

    with DDGS() as ddgs:
        raw_results = ddgs.text(query, max_results=top_k)

    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in raw_results
    ]


def _search_web_handler(payload: WebSearchInput) -> WebSearchOutput:
    results = [
        SearchResult(**res)
        for res in search_web(
            payload.query,
            payload.top_k,
        )
    ]
    return WebSearchOutput(results=results)


register_tool(
    ToolSpec(
        name="search.web",
        input_model=WebSearchInput,
        output_model=WebSearchOutput,
        handler=_search_web_handler,
    )
)

__all__ = [
    "WebSearchInput",
    "SearchResult",
    "WebSearchOutput",
    "search_web",
]
