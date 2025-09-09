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


# Stub extract module required by fetch_gdelt
extract_stub = types.ModuleType("sentimental_cap_predictor.news.extract")
extract_stub.fetch_html = lambda url, timeout=20: ""  # noqa: E731
extract_stub.extract_main_text = lambda html, url=None: ""  # noqa: E731
sys.modules["sentimental_cap_predictor.news.extract"] = extract_stub
news_pkg.extract = extract_stub


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

