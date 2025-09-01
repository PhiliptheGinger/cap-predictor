import pandas as pd
import pytest
import requests

from sentimental_cap_predictor.data import news
from sentimental_cap_predictor.data.news import (
    FileSource,
    GDELTSource,
    fetch_headline,
    fetch_news,
)


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
        "http://example.com", use_headless=use_headless
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


def test_query_gdelt_for_news_handles_timeout(monkeypatch):
    def fake_get(url, params, timeout):  # noqa: ANN001
        raise requests.Timeout()

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(requests.Timeout):
        news.query_gdelt_for_news(
            query="NVDA", start_date="20240101000000", end_date="20240102000000"
        )


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

    monkeypatch.setattr(news, "GDELTSource", lambda max_records=1: DummySource())
    headline = news.fetch_headline("NVDA")
    assert headline == "Example"


def test_fetch_first_gdelt_article_prefers_content(monkeypatch):
    df = pd.DataFrame([{"title": "Headline", "url": "http://example.com"}])

    monkeypatch.setattr(news, "query_gdelt_for_news", lambda q, s, e: df)
    monkeypatch.setattr(news, "extract_article_content", lambda url: "Body text")

    article = news.fetch_first_gdelt_article("NVDA")
    assert article.content == "Body text"
