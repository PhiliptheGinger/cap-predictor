import importlib.util
import sys
from pathlib import Path
import importlib.util

import httpx

# Load config and gdelt_client without importing heavy modules
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

GdeltClient = gdelt_client.GdeltClient
search_gdelt = gdelt_client.search_gdelt
GDELT_API_URL = config.GDELT_API_URL


def test_gdelt_client_search(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "Example",
                "url": "http://ex",
                "domain": "ex",
                "source": "Feed",
                "seendate": "2024-01-01T00:00:00Z",
            }
        ]
    }

    class DummyResponse:
        status_code = 200

        def json(self):
            return payload

        def raise_for_status(self):
            pass

    captured = {}

    def fake_get(self, url, params):  # noqa: ANN001
        captured["url"] = url
        captured["params"] = params
        return DummyResponse()

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    client = GdeltClient()
    stubs = client.search("nvda")
    assert stubs[0].url == "http://ex"
    assert captured["url"] == GDELT_API_URL
    assert captured["params"]["maxrecords"] == 75


def test_search_gdelt_wrapper(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "Example",
                "url": "http://ex",
                "domain": "ex",
                "source": "Feed",
                "seendate": "2024-01-01T00:00:00Z",
            }
        ]
    }

    class DummyResponse:
        status_code = 200

        def json(self):
            return payload

        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, params: DummyResponse())

    results = search_gdelt("nvda", max_results=1)
    assert results[0]["url"] == "http://ex"
