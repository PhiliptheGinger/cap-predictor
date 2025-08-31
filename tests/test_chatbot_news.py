import json
import sys
import types
import pandas as pd
import requests

# Stub heavy dependencies used by package initialization
sys.modules.setdefault(
    "colorama",
    types.SimpleNamespace(
        Fore=types.SimpleNamespace(RED="", GREEN="", YELLOW=""),
        Style=types.SimpleNamespace(RESET_ALL=""),
        init=lambda *a, **k: None,
    ),
)
logger_stub = types.SimpleNamespace(
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
    exception=lambda *a, **k: None,
    remove=lambda *a, **k: None,
    add=lambda *a, **k: 0,
)
sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logger_stub))
sys.modules.setdefault("typer", types.SimpleNamespace(Typer=lambda *a, **k: None))

# Provide a minimal dataset module so the frontend can import it lazily
calls: list[str] = []


def query_gdelt_for_news(query, start_date, end_date):  # noqa: ANN001
    calls.append(query)
    resp = requests.get("http://example.com", params={"query": query}, timeout=30)
    articles = resp.json()["articles"]
    return pd.DataFrame(articles)


dataset_stub = types.SimpleNamespace(query_gdelt_for_news=query_gdelt_for_news)
sys.modules["sentimental_cap_predictor.dataset"] = dataset_stub

from sentimental_cap_predictor import chatbot_frontend


def test_fetch_gdelt_news_uses_helper(monkeypatch):
    payload = {
        "articles": [
            {
                "seendate": "2024-01-01T00:00:00Z",
                "title": "Example headline",
                "source": "Feed",
            }
        ]
    }

    class DummyResponse:
        def json(self):
            return payload

        def raise_for_status(self):  # pragma: no cover - no-op
            pass

    def fake_get(url, params, timeout):  # noqa: ANN001
        assert params["query"] == "NVDA"
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    out = chatbot_frontend.fetch_gdelt_news("NVDA")
    data = json.loads(out)

    assert data[0]["title"] == "Example headline"
    assert calls == ["NVDA"]
