"""Utilities for interacting with news APIs."""

from . import store
from .extractor import ArticleExtractor, ExtractedArticle
from .fetcher import HtmlFetcher
from .gdelt_client import ArticleStub, GdeltClient, search_gdelt
from .scoring import score_news

__all__ = [
    "ArticleStub",
    "GdeltClient",
    "search_gdelt",
    "HtmlFetcher",
    "ArticleExtractor",
    "ExtractedArticle",
    "store",
    "score_news",
]
