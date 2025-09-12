import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import httpx

# Create lightweight package structure to satisfy relative imports
root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("sentimental_cap_predictor")
news_pkg = types.ModuleType("sentimental_cap_predictor.news")
news_pkg.__path__ = [str(root / "src" / "sentimental_cap_predictor" / "news")]
sys.modules.setdefault("sentimental_cap_predictor", pkg)
sys.modules.setdefault("sentimental_cap_predictor.news", news_pkg)
pkg.news = news_pkg

# Load fetcher module from source file
spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.fetcher",
    root / "src" / "sentimental_cap_predictor" / "news" / "fetcher.py",
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

    async def fake_sleep(delay):  # noqa: ANN001
        return None

    def fake_random():  # noqa: ANN001
        return 0.0

    called: dict[str, tuple[str, str, str]] = {}

    def fake_log_error(url, stage, message):  # noqa: ANN001
        called["args"] = (url, stage, message)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(fetcher_mod, "log_error", fake_log_error)
    monkeypatch.setattr(fetcher_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(fetcher_mod.random, "random", fake_random)

    fetcher = HtmlFetcher()
    html = run(fetcher.get("http://e", max_retries=2))
    assert html is None
    assert called["args"][0] == "http://e"
    assert called["args"][1] == "fetch"
    run(fetcher.aclose())


def test_playwright_fallback(monkeypatch):
    class DummyResponse:
        status_code = 403
        text = "blocked"

    async def fake_get(self, url, follow_redirects=True):  # noqa: ANN001
        return DummyResponse()

    async def fake_pw(url, timeout_s):  # noqa: ANN001
        return "browser"

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    monkeypatch.setattr(fetcher_mod, "async_playwright", object())
    monkeypatch.setattr(fetcher_mod, "_fetch_with_playwright", fake_pw)
    monkeypatch.setenv("NEWS_USE_PLAYWRIGHT", "1")

    fetcher = HtmlFetcher()
    html = run(fetcher.get("http://blocked"))
    assert html == "browser"
    run(fetcher.aclose())
