import importlib.util
import logging
import sys
import types
from pathlib import Path


# Set up lightweight package structure to avoid heavy imports
pkg = types.ModuleType("sentimental_cap_predictor")
pkg.__path__ = []
news_pkg = types.ModuleType("sentimental_cap_predictor.news")
news_pkg.__path__ = []
memory_pkg = types.ModuleType("sentimental_cap_predictor.memory")
memory_pkg.__path__ = []
sys.modules.setdefault("sentimental_cap_predictor", pkg)
sys.modules.setdefault("sentimental_cap_predictor.news", news_pkg)
sys.modules.setdefault("sentimental_cap_predictor.memory", memory_pkg)
pkg.news = news_pkg
pkg.memory = memory_pkg


# Minimal stubs for modules imported by fetch_gdelt
gdelt_stub = types.ModuleType("sentimental_cap_predictor.news.gdelt_client")

class DummyClient:
    def search(self, query, max_records=15):  # noqa: ANN001
        return []


gdelt_stub.GdeltClient = DummyClient
gdelt_stub.ArticleStub = types.SimpleNamespace
sys.modules["sentimental_cap_predictor.news.gdelt_client"] = gdelt_stub
news_pkg.gdelt_client = gdelt_stub

fetcher_stub = types.ModuleType("sentimental_cap_predictor.news.fetcher")

class DummyFetcher:
    async def get(self, url, max_retries=3):  # noqa: ANN001
        return ""

    async def aclose(self):  # pragma: no cover - no-op
        pass


fetcher_stub.HtmlFetcher = DummyFetcher
sys.modules["sentimental_cap_predictor.news.fetcher"] = fetcher_stub
news_pkg.fetcher = fetcher_stub

extractor_stub = types.ModuleType("sentimental_cap_predictor.news.extractor")

class DummyExtractor:
    def extract(self, html, url=None):  # noqa: ANN001
        return types.SimpleNamespace(text="")


extractor_stub.ArticleExtractor = DummyExtractor
extractor_stub.ExtractedArticle = types.SimpleNamespace
sys.modules["sentimental_cap_predictor.news.extractor"] = extractor_stub
news_pkg.extractor = extractor_stub


# Load vector_store module
root = Path(__file__).resolve().parents[1]
vs_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.memory.vector_store",
    root / "src" / "sentimental_cap_predictor" / "memory" / "vector_store.py",
)
vector_store = importlib.util.module_from_spec(vs_spec)
sys.modules["sentimental_cap_predictor.memory.vector_store"] = vector_store
vs_spec.loader.exec_module(vector_store)
memory_pkg.vector_store = vector_store


# Load fetch_gdelt module
fg_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.fetch_gdelt",
    root / "src" / "sentimental_cap_predictor" / "news" / "fetch_gdelt.py",
)
fetch_gdelt = importlib.util.module_from_spec(fg_spec)
sys.modules["sentimental_cap_predictor.news.fetch_gdelt"] = fetch_gdelt
fg_spec.loader.exec_module(fetch_gdelt)
news_pkg.fetch_gdelt = fetch_gdelt


def test_store_chunks_upserts(monkeypatch):
    calls = []

    def fake_upsert(doc_id, text, metadata):  # noqa: ANN001
        calls.append((doc_id, text, metadata))

    monkeypatch.setattr(fetch_gdelt.vector_store, "upsert", fake_upsert)

    text = "x" * 2500
    result = {
        "text": text,
        "title": "Title",
        "url": "http://example.com/article",
        "seendate": "20240101",
        "domain": "example.com",
    }

    fetch_gdelt._store_chunks(result)
    assert len(calls) == 3
    for idx, call in enumerate(calls):
        doc_id, chunk, metadata = call
        assert doc_id == f"http://example.com/article#{idx}"
        assert metadata["title"] == "Title"
        assert metadata["url"] == "http://example.com/article"
        assert metadata["seendate"] == "20240101"
        assert metadata["domain"] == "example.com"
        assert len(chunk) <= 1000


def test_store_chunks_handles_errors(monkeypatch, caplog):
    def boom(doc_id, text, metadata):  # noqa: ANN001
        raise RuntimeError("fail")

    monkeypatch.setattr(fetch_gdelt.vector_store, "upsert", boom)
    result = {
        "text": "body",
        "title": "T",
        "url": "http://e",
        "seendate": "S",
        "domain": "D",
    }
    with caplog.at_level(logging.WARNING):
        fetch_gdelt._store_chunks(result)
    assert any("Vector store upsert failed" in m for m in caplog.messages)


def test_search_gdelt_skips_blocked_domains(monkeypatch, caplog):
    def fake_search(self, query, max_records=15):  # noqa: ANN001
        return [
            types.SimpleNamespace(url="http://wsj.com/a", domain="wsj.com", title="A", seendate=None),
            types.SimpleNamespace(
                url="http://reuters.com/b", domain="reuters.com", title="B", seendate=None
            ),
        ]


    monkeypatch.setattr(fetch_gdelt.GdeltClient, "search", fake_search)

    calls: list[str] = []

    def fake_fetch_html(url):  # noqa: ANN001
        calls.append(url)
        return "<html>body</html>"

    monkeypatch.setattr(fetch_gdelt, "fetch_html", fake_fetch_html)
    monkeypatch.setenv("NEWS_BLOCKED_DOMAINS", "wsj.com")

    with caplog.at_level(logging.INFO):
        articles = fetch_gdelt.search_gdelt("q", max_records=2)
    assert len(articles) == 1
    assert calls == ["http://reuters.com/b"]
    assert any("wsj.com" in m for m in caplog.messages)

