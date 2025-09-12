import importlib.util
import sys
from pathlib import Path

import trafilatura

# Load extractor module
spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.extractor",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "news"
    / "extractor.py",
)
extractor_mod = importlib.util.module_from_spec(spec)
sys.modules["sentimental_cap_predictor.news.extractor"] = extractor_mod
spec.loader.exec_module(extractor_mod)

ArticleExtractor = extractor_mod.ArticleExtractor
ExtractedArticle = extractor_mod.ExtractedArticle


def test_extractor_basic():
    html = "<html><body><p>Hello</p></body></html>"
    extractor = ArticleExtractor()
    result = extractor.extract(html, url="http://e")
    assert isinstance(result, ExtractedArticle)
    assert "Hello" in result.text


def test_extractor_fallback(monkeypatch):
    monkeypatch.setattr(trafilatura, "extract", lambda *a, **k: None)

    class DummyDoc:
        def __init__(self, html):
            self._html = html

        def summary(self, html_partial=True):  # noqa: ANN001
            return "<p>Fallback</p>"

        def title(self):  # noqa: ANN001
            return "Title"

    monkeypatch.setattr(extractor_mod, "Document", DummyDoc)

    extractor = ArticleExtractor()
    result = extractor.extract("<html><body></body></html>")
    assert result and "Fallback" in result.text
