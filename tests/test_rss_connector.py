import importlib.util
import sys
import types
from pathlib import Path

import requests

dummy_pkg = types.ModuleType("sentimental_cap_predictor")
dummy_pkg.__path__ = []  # type: ignore[attr-defined]
dummy_llm_pkg = types.ModuleType("sentimental_cap_predictor.llm_core")
dummy_llm_pkg.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("sentimental_cap_predictor", dummy_pkg)
sys.modules.setdefault("sentimental_cap_predictor.llm_core", dummy_llm_pkg)

spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.llm_core.connectors.rss_connector",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "llm_core"
    / "connectors"
    / "rss_connector.py",
)
rss_connector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rss_connector
spec.loader.exec_module(rss_connector)


RSS_TEXT = """<?xml version='1.0'?>
<rss version='2.0'>
<channel>
<title>Sample</title>
<item>
<title>First</title>
<link>http://example.com/1</link>
<description>Hello</description>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
</item>
</channel>
</rss>"""


def test_load_feed_list(tmp_path: Path) -> None:
    path = tmp_path / "feeds.txt"
    path.write_text("http://example.com/rss\n#comment\nhttp://example.com/2\n")
    urls = rss_connector.load_feed_list(path)
    assert urls == ["http://example.com/rss", "http://example.com/2"]


def test_fetch_feeds(monkeypatch):
    class DummyResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):  # noqa: ANN001
        return DummyResponse(RSS_TEXT)

    monkeypatch.setattr(requests, "get", fake_get)
    articles = rss_connector.fetch_feeds(["http://example.com/rss"])
    assert len(articles) == 1
    assert articles[0]["title"] == "First"


def test_fetch_all_with_newsapi(monkeypatch):
    class DummyResponse:
        def __init__(self, text: str | None = None, payload: dict | None = None):
            self.text = text
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, params=None, timeout=30):  # noqa: ANN001
        if url == "http://example.com/rss":
            return DummyResponse(text=RSS_TEXT)
        assert params["apiKey"] == "KEY"
        return DummyResponse(payload={
            "articles": [
                {
                    "title": "NewsAPI",
                    "url": "http://newsapi.org/1",
                    "description": "Desc",
                    "publishedAt": "2024-01-01T00:00:00Z",
                }
            ]
        })

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("NEWSAPI_API_KEY", "KEY")
    articles = rss_connector.fetch_all(["http://example.com/rss"], query="test")
    titles = {a["title"] for a in articles}
    assert "NewsAPI" in titles
    monkeypatch.delenv("NEWSAPI_API_KEY")
