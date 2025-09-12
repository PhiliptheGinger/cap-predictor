import importlib.util
import json
import sys
import types
from pathlib import Path

from typer.testing import CliRunner


# Load CLI module without importing the full package
dummy_pkg = types.ModuleType("sentimental_cap_predictor")
dummy_pkg.__path__ = []
sys.modules.setdefault("sentimental_cap_predictor", dummy_pkg)

dummy_news_pkg = types.ModuleType("sentimental_cap_predictor.news")
dummy_news_pkg.__path__ = [
    str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "sentimental_cap_predictor"
        / "news"
    )
]
sys.modules.setdefault("sentimental_cap_predictor.news", dummy_news_pkg)
dummy_pkg.news = dummy_news_pkg

# Provide lightweight stubs for modules with heavy dependencies
dummy_config = types.ModuleType("sentimental_cap_predictor.config")
dummy_config.GDELT_API_URL = "https://example.com"
sys.modules.setdefault("sentimental_cap_predictor.config", dummy_config)
dummy_pkg.config = dummy_config

extractor_mod = types.ModuleType("sentimental_cap_predictor.news.extractor")


class DummyExtracted:
    def __init__(self, text: str = "body"):
        self.text = text
        self.title = "T"
        self.byline = None
        self.date = None


class DummyExtractor:
    def extract(self, html, url=None):  # noqa: ANN001
        return DummyExtracted()


extractor_mod.ArticleExtractor = DummyExtractor
extractor_mod.ExtractedArticle = DummyExtracted
sys.modules.setdefault("sentimental_cap_predictor.news.extractor", extractor_mod)
dummy_news_pkg.extractor = extractor_mod

fetcher_mod = types.ModuleType("sentimental_cap_predictor.news.fetcher")


class DummyFetcher:
    async def get(self, url, *, max_retries=3):  # noqa: ANN001
        return "<html></html>"

    async def aclose(self):
        return None


fetcher_mod.HtmlFetcher = DummyFetcher
sys.modules.setdefault("sentimental_cap_predictor.news.fetcher", fetcher_mod)
dummy_news_pkg.fetcher = fetcher_mod

gdelt_mod = types.ModuleType("sentimental_cap_predictor.news.gdelt_client")


class DummyStub:
    def __init__(
        self,
        url: str = "http://e",
        title: str = "T",
        domain: str = "d",
        seendate=None,
        source: str | None = None,
    ):
        self.url = url
        self.title = title
        self.domain = domain
        self.seendate = seendate
        self.source = source


class DummyClient:
    def search(self, query, max_records):  # noqa: ANN001
        return [DummyStub()]


gdelt_mod.GdeltClient = DummyClient
gdelt_mod.ArticleStub = DummyStub
sys.modules.setdefault("sentimental_cap_predictor.news.gdelt_client", gdelt_mod)
dummy_news_pkg.gdelt_client = gdelt_mod

module_name = "sentimental_cap_predictor.news.cli"
spec = importlib.util.spec_from_file_location(
    module_name,
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "news"
    / "cli.py",
)
cli = importlib.util.module_from_spec(spec)
sys.modules[module_name] = cli
spec.loader.exec_module(cli)
dummy_news_pkg.cli = cli

app = cli.app


def test_search_cli(monkeypatch):
    runner = CliRunner()

    from sentimental_cap_predictor.news.gdelt_client import ArticleStub

    def fake_search(self, query, max_records):  # noqa: ANN001
        assert query == "NVDA"
        assert max_records == 2
        return [ArticleStub(url="http://e", title="T", domain="d", seendate=None)]

    monkeypatch.setattr(cli.GdeltClient, "search", fake_search)

    result = runner.invoke(
        app,
        ["search", "--query", "NVDA", "--max", "2"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data[0]["url"] == "http://e"


def test_ingest_cli(monkeypatch):
    runner = CliRunner()

    from sentimental_cap_predictor.news.gdelt_client import ArticleStub

    monkeypatch.setattr(
        cli.GdeltClient,
        "search",
        lambda self, query, max_records: [
            ArticleStub(url="http://e", title="T", domain="d", seendate=None)
        ],
    )
    monkeypatch.setattr(cli, "fetch_html", lambda url: "<html></html>")

    class DummyArt:
        text = "body"
        title = "T"
        byline = None
        date = None

    monkeypatch.setattr(
        cli.ArticleExtractor, "extract", lambda self, html, url=None: DummyArt()
    )

    stored = {"articles": [], "contents": []}
    monkeypatch.setattr(
        cli.store,
        "upsert_article",
        lambda data: stored["articles"].append(data),
    )
    monkeypatch.setattr(
        cli.store,
        "upsert_content",
        lambda url, text, summary=None, sentiment=None, relevance=None: stored[
            "contents"
        ].append((url, text)),
    )

    result = runner.invoke(
        app,
        ["ingest", "--query", "NVDA"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert stored["articles"][0]["url"] == "http://e"
    assert stored["contents"][0][1] == "body"


def test_score_cli(monkeypatch, tmp_path):
    runner = CliRunner()
    import sqlite3

    db_path = tmp_path / "news.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE articles (url TEXT PRIMARY KEY, seendate TEXT)")
    conn.execute(
        "CREATE TABLE contents (url TEXT PRIMARY KEY, text TEXT, relevance REAL)"
    )
    conn.execute("INSERT INTO articles (url, seendate) VALUES ('u', '2024-01-01')")
    conn.execute(
        "INSERT INTO contents (url, text, relevance) VALUES ('u', 'hello', 0.5)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(cli.store, "DB_PATH", db_path)

    called: dict = {}

    def fake_score_news(df, **kwargs):  # noqa: ANN001
        called["df"] = df
        df = df.copy()
        df["score"] = [1.0]
        df["ewma_score"] = [1.0]
        return df

    monkeypatch.setattr(cli, "score_news", fake_score_news)

    result = runner.invoke(app, ["score"], catch_exceptions=False)
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data[0]["score"] == 1.0
    assert "df" in called
    assert int(called["df"]["length"].iloc[0]) == len("hello")

