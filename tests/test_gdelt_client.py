import importlib.util
import sys
import types
from pathlib import Path

import requests

# Create lightweight package stubs to avoid heavy imports
# from sentimental_cap_predictor
pkg = types.ModuleType("sentimental_cap_predictor")
pkg.__path__ = []
news_pkg = types.ModuleType("sentimental_cap_predictor.news")
news_pkg.__path__ = []
sys.modules.setdefault("sentimental_cap_predictor", pkg)
sys.modules.setdefault("sentimental_cap_predictor.news", news_pkg)

# Load configuration module
config_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.config",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "config.py",
)
config = importlib.util.module_from_spec(config_spec)
sys.modules["sentimental_cap_predictor.config"] = config
config_spec.loader.exec_module(config)

# Load GDELT client module
client_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.gdelt_client",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "news"
    / "gdelt_client.py",
)
gdelt_client = importlib.util.module_from_spec(client_spec)
sys.modules["sentimental_cap_predictor.news.gdelt_client"] = gdelt_client
client_spec.loader.exec_module(gdelt_client)

search_gdelt = gdelt_client.search_gdelt
GDELT_API_URL = config.GDELT_API_URL


def test_search_gdelt_parses_response(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "Example",
                "url": "http://ex",
                "source": "Feed",
                "seendate": "20240101",
            }
        ]
    }

    class DummyResponse:
        def json(self):
            return payload

        def raise_for_status(self):  # pragma: no cover - no-op
            pass

    captured = {}

    def fake_get(url, params, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["params"] = params
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    results = search_gdelt("nvda")
    assert results == [
        {
            "title": "Example",
            "url": "http://ex",
            "source": "Feed",
            "pubdate": "20240101",
        }
    ]
    assert captured["url"] == GDELT_API_URL
    assert captured["params"]["maxrecords"] == 3


def test_search_gdelt_default_limit(monkeypatch):
    payload = {
        "articles": [
            {"title": str(i), "url": str(i), "source": "s", "seendate": "d"}
            for i in range(5)
        ]
    }

    class DummyResponse:
        def json(self):
            return payload

        def raise_for_status(self):  # pragma: no cover - no-op
            pass

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, params, timeout: DummyResponse(),  # noqa: ANN001
    )

    results = search_gdelt("test")
    assert len(results) == 3


def test_search_gdelt_handles_errors(monkeypatch):
    def fake_get(url, params, timeout):  # noqa: ANN001
        raise requests.RequestException

    monkeypatch.setattr(requests, "get", fake_get)
    assert search_gdelt("nvda") == []
