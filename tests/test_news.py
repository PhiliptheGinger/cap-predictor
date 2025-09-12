import importlib.util
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

dummy_pkg = types.ModuleType("sentimental_cap_predictor")
dummy_pkg.__path__ = []
dummy_data_pkg = types.ModuleType("sentimental_cap_predictor.data")
dummy_data_pkg.__path__ = []
sys.modules.setdefault("sentimental_cap_predictor", dummy_pkg)
sys.modules.setdefault("sentimental_cap_predictor.data", dummy_data_pkg)

module_name = "sentimental_cap_predictor.data.news"
spec = importlib.util.spec_from_file_location(
    module_name,
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "data"
    / "news.py",
)
news = importlib.util.module_from_spec(spec)
sys.modules[module_name] = news
spec.loader.exec_module(news)

FileSource = news.FileSource
GDELTSource = news.GDELTSource
fetch_headline = news.fetch_headline
fetch_news = news.fetch_news


def test_fetch_news_returns_columns():
    df = fetch_news(query="NVDA")
    assert list(df.columns) == ["date", "headline", "source"]


def test_fetch_news_ticker_keyword_backwards_compatibility():
    df = fetch_news(ticker="NVDA")
    assert list(df.columns) == ["date", "headline", "source"]


def test_file_source_reads_csv(tmp_path):
    csv_path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "headline": ["Example"],
            "source": ["Test"],
            "ticker": ["NVDA"],
        }
    ).to_csv(csv_path, index=False)

    source = FileSource(csv_path)
    df = source.fetch(query="NVDA")
    assert len(df) == 1
    assert df.iloc[0]["headline"] == "Example"
    assert list(df.columns) == ["date", "headline", "source"]


def test_gdelt_source_fetch(monkeypatch):
    payload = {
        "articles": [
            {
                "seendate": "2024-01-01T00:00:00Z",
                "title": "Example",
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

    source = GDELTSource(days=1, max_records=1)
    df = source.fetch(query="NVDA")
    assert list(df.columns) == ["date", "headline", "source"]
    assert df.iloc[0]["headline"] == "Example"
    assert df.iloc[0]["source"] == "Feed"


@pytest.mark.parametrize(
    "use_headless, expected",
    [(True, "HeadlessChrome"), (False, "Chrome/58")],
)
def test_extract_article_content_user_agent(
    monkeypatch,
    use_headless,
    expected,
):
    captured = {}

    class DummyArticle:
        def __init__(self, url, config, keep_article_html=True):
            captured["ua"] = config.browser_user_agent
            captured["timeout"] = config.request_timeout
            self.text = "body"

        def download(self):
            pass

        def parse(self):
            pass

    monkeypatch.setattr(news, "Article", DummyArticle)

    text = news.extract_article_content(
        "http://example.com",
        use_headless=use_headless,
    )
    assert text == "body"
    assert expected in captured["ua"]
    assert captured["timeout"] == 10


def test_query_gdelt_for_news_uses_timeout(monkeypatch):
    captured = {}

    class DummyResponse:
        def json(self):
            return {"articles": []}

        def raise_for_status(self):  # pragma: no cover - no-op
            pass

    def fake_get(url, params, timeout):  # noqa: ANN001
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    df = news.query_gdelt_for_news(
        query="NVDA", start_date="20240101000000", end_date="20240102000000"
    )
    assert captured["timeout"] == 30
    assert isinstance(df, pd.DataFrame)


def test_query_gdelt_for_news_appends_language(monkeypatch):
    captured = {}

    class DummyResponse:
        def json(self):  # pragma: no cover - trivial
            return {"articles": []}

        def raise_for_status(self):  # pragma: no cover - no-op
            pass

    def fake_get(url, params, timeout):  # noqa: ANN001
        captured["query"] = params["query"]
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    news.query_gdelt_for_news(
        query="NVDA",
        start_date="20240101000000",
        end_date="20240102000000",
        language="english",
    )
    assert captured["query"] == "NVDA lang:english"


def test_query_gdelt_for_news_handles_timeout(monkeypatch):
    def fake_get(url, params, timeout):  # noqa: ANN001
        raise requests.Timeout()

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.Timeout):
        news.query_gdelt_for_news(
            query="NVDA",
            start_date="20240101000000",
            end_date="20240102000000",
        )


def test_query_gdelt_for_news_non_json(monkeypatch):
    class DummyResponse:
        text = "Not JSON"

        def json(self):  # pragma: no cover - executed in test
            raise ValueError("no json")

        def raise_for_status(self):  # pragma: no cover - no-op
            pass

    def fake_get(url, params, timeout):  # noqa: ANN001
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        news.query_gdelt_for_news(
            query="NVDA",
            start_date="20240101000000",
            end_date="20240102000000",
        )

    msg = str(excinfo.value)
    assert "Failed to parse GDELT response as JSON" in msg
    assert "Not JSON" in msg


def test_try_to_extract_filters_login_page(monkeypatch):
    calls = []

    def fake_extract(url, use_headless=False):  # noqa: ANN001
        calls.append(use_headless)
        return "Please sign in to continue"

    monkeypatch.setattr(news, "extract_article_content", fake_extract)

    text = news.try_to_extract("http://example.com")
    assert text is None
    assert calls == [False, True]


def test_fetch_headline_uses_gdelt_source(monkeypatch):
    class DummySource:
        def fetch(self, query):  # noqa: ANN001
            return pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-01")],
                    "headline": ["Example"],
                    "source": ["Feed"],
                }
            )

    monkeypatch.setattr(
        news,
        "GDELTSource",
        lambda max_records=1: DummySource(),
    )
    headline = news.fetch_headline("NVDA")
    assert headline == "Example"


def test_fetch_article_prefers_content(monkeypatch):
    df = pd.DataFrame([{"title": "Headline", "url": "http://example.com"}])

    monkeypatch.setattr(
        news,
        "query_gdelt_for_news",
        lambda q, s, e, *, max_records=100, language=None: df,
    )
    monkeypatch.setattr(
        news,
        "extract_article_content",
        lambda *a, **k: "Body text",
    )

    spec = news.FetchArticleSpec(query="NVDA")
    article = news.fetch_article(spec)
    assert article.content == "Body text"


def test_fetch_article_raises_on_missing_content(monkeypatch):
    df = pd.DataFrame([{"title": "Headline", "url": "http://example.com"}])

    monkeypatch.setattr(
        news,
        "query_gdelt_for_news",
        lambda q, s, e, *, max_records=100, language=None: df,
    )
    monkeypatch.setattr(
        news,
        "extract_article_content",
        lambda *a, **k: None,
    )

    spec = news.FetchArticleSpec(query="NVDA")
    with pytest.raises(RuntimeError):
        news.fetch_article(spec)


def test_fetch_article_custom_window_and_limit(monkeypatch):
    df = pd.DataFrame([{"title": "Headline", "url": "http://example.com"}])
    captured: dict[str, str | int] = {}

    def fake_query(q, s, e, *, max_records, language=None):  # noqa: ANN001
        captured["start"] = s
        captured["end"] = e
        captured["max"] = max_records
        captured["language"] = language
        return df

    monkeypatch.setattr(news, "query_gdelt_for_news", fake_query)
    monkeypatch.setattr(
        news,
        "extract_article_content",
        lambda *a, **k: "Body",
    )

    class DummyDateTime(datetime):
        @classmethod
        def utcnow(cls):  # noqa: D401
            return cls(2024, 1, 10, 12, 0, 0)

    monkeypatch.setattr(news, "datetime", DummyDateTime)

    spec = news.FetchArticleSpec(query="NVDA", days=3, max_records=5)
    article = news.fetch_article(spec)
    assert article.title == "Headline"
    assert captured["start"] == "20240107120000"
    assert captured["end"] == "20240110120000"
    assert captured["max"] == 5
    assert captured["language"] == "english"


def test_fetch_article_applies_filters_and_novelty(monkeypatch):
    df = pd.DataFrame(
        [
            {"title": "Keyword match", "url": "http://known.com/a"},
            {"title": "Keyword match", "url": "http://new.com/b"},
            {"title": "Other", "url": "http://bad.com/c"},
        ]
    )

    monkeypatch.setattr(
        news,
        "query_gdelt_for_news",
        lambda q, s, e, *, max_records=100, language=None: df,
    )
    monkeypatch.setattr(
        news, "extract_article_content", lambda url, use_headless=False: "text"
    )

    spec = news.FetchArticleSpec(
        query="q",
        must_contain_any=("keyword",),
        avoid_domains=("bad.com",),
        novelty_against_urls=("http://known.com/a",),
    )
    article = news.fetch_article(spec)
    assert article.url == "http://new.com/b"


def test_fetch_article_title_novelty(monkeypatch):
    df = pd.DataFrame(
        [
            {"title": "Seen headline", "url": "http://old.com/a"},
            {"title": "Fresh perspective", "url": "http://new.com/b"},
        ]
    )

    monkeypatch.setattr(
        news,
        "query_gdelt_for_news",
        lambda q, s, e, *, max_records=100, language=None: df,
    )
    monkeypatch.setattr(
        news, "extract_article_content", lambda url, use_headless=False: "text"
    )

    spec = news.FetchArticleSpec(query="q")
    article = news.fetch_article(spec, seen_titles=("Seen headline",))
    assert article.url == "http://new.com/b"


def test_fetch_article_skips_non_english(monkeypatch):
    df = pd.DataFrame(
        [
            {"title": "Titulo", "url": "http://es.com", "language": "spanish"},
            {
                "title": "Headline",
                "url": "http://en.com",
                "language": "english",
            },
        ]
    )

    def fake_query(q, s, e, *, max_records=100, language=None):  # noqa: ANN001
        return df

    extracted: list[str] = []

    def fake_extract(url, use_headless=False):  # noqa: ANN001, D401
        extracted.append(url)
        return "body"

    monkeypatch.setattr(news, "query_gdelt_for_news", fake_query)
    monkeypatch.setattr(news, "extract_article_content", fake_extract)

    spec = news.FetchArticleSpec(query="q")
    article = news.fetch_article(spec)
    assert article.url == "http://en.com"
    assert extracted == ["http://en.com"]


@pytest.mark.parametrize(
    "spec_lang, article_lang",
    [("english", "en"), ("en", "english")],
)
def test_is_valid_candidate_accepts_language_variants(spec_lang, article_lang):
    article = pd.Series({"title": "Headline", "url": "http://example.com", "language": article_lang})
    spec = news.FetchArticleSpec(query="q", language=spec_lang)
    assert news.is_valid_candidate(article, spec)
