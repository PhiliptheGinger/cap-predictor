import pandas as pd

from sentimental_cap_predictor.data.news import FileSource, fetch_news
from sentimental_cap_predictor import dataset
import pytest


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


@pytest.mark.parametrize("use_headless, expected", [(True, "HeadlessChrome"), (False, "Chrome/58")])
def test_extract_article_content_user_agent(monkeypatch, use_headless, expected):
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

    text = dataset.extract_article_content("http://example.com", use_headless=use_headless)
    assert text == "body"
    assert expected in captured["ua"]
    assert captured["timeout"] == 10
