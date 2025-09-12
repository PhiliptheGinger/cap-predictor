"""Utilities for interacting with news APIs."""

from .gdelt_client import ArticleStub, GdeltClient, search_gdelt
from .fetcher import HtmlFetcher
from .extractor import ArticleExtractor, ExtractedArticle
from . import store

__all__ = [
    "ArticleStub",
    "GdeltClient",
    "search_gdelt",
    "HtmlFetcher",
    "ArticleExtractor",
    "ExtractedArticle",
    "store",
]
