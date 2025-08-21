import pandas as pd
import pytest
import requests

from sentimental_cap_predictor import dataset
from sentimental_cap_predictor.data.news import (
    FileSource,
    GDELTSource,
    fetch_news,
)


def test_fetch_news_returns_columns():
    df = fetch_news("NVDA")
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
    df = source.fetch("NVDA")
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
    df = source.fetch("NVDA")
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

    monkeypatch.setattr(dataset, "Article", DummyArticle)

    text = dataset.extract_article_content(
        "http://example.com", use_headless=use_headless
    )
    assert text == "body"
    assert expected in captured["ua"]
    assert captured["timeout"] == 10
