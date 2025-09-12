import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx

# Load fetcher module
spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.fetcher",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "news"
    / "fetcher.py",
)
fetcher_mod = importlib.util.module_from_spec(spec)
sys.modules["sentimental_cap_predictor.news.fetcher"] = fetcher_mod
spec.loader.exec_module(fetcher_mod)

HtmlFetcher = fetcher_mod.HtmlFetcher


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_fetcher_success(monkeypatch):
    class DummyResponse:
        status_code = 200
        text = "ok"

    async def fake_get(self, url, follow_redirects=True):  # noqa: ANN001
        return DummyResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    fetcher = HtmlFetcher()
    html = run(fetcher.get("http://e"))
    assert html == "ok"
    run(fetcher.aclose())


def test_fetcher_failure(monkeypatch):
    async def fake_get(self, url, follow_redirects=True):  # noqa: ANN001
        raise httpx.ProxyError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    fetcher = HtmlFetcher()
    html = run(fetcher.get("http://e", max_retries=2))
    assert html is None
    run(fetcher.aclose())
