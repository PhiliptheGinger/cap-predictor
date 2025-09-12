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

dummy_config = types.ModuleType("sentimental_cap_predictor.config")
dummy_config.GDELT_API_URL = "https://example.com"
sys.modules.setdefault("sentimental_cap_predictor.config", dummy_config)
dummy_pkg.config = dummy_config

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


def test_fetch_gdelt_cli(monkeypatch):
    runner = CliRunner()

    def fake_search_gdelt(query, max_results):  # noqa: ANN001
        assert query == "NVDA"
        assert max_results == 2
        return [
            {
                "title": "Example",
                "url": "http://e",
                "source": "Feed",
                "pubdate": "2024",
            }
        ]

    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.search_gdelt",
        fake_search_gdelt,
    )

    result = runner.invoke(
        app,
        ["fetch-gdelt", "--query", "NVDA", "--max", "2"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data[0]["title"] == "Example"


def test_read_cli(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.fetch_html",
        lambda url: "<html></html>",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.extract_main",
        lambda html, url=None: "body",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.analyze_text",
        lambda text: {"lang": "en", "word_count": 1},
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.summarize_text",
        lambda text: "summary",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.chunk_text",
        lambda text, max_tokens, overlap: ["chunk"],
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.translate_text",
        lambda text, target_lang: "translated",
    )

    result = runner.invoke(
        app,
        [
            "read",
            "--url",
            "http://example.com",
            "--summarize",
            "--analyze",
            "--chunks",
            "5",
            "--overlap",
            "1",
            "--translate",
            "en",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data["text"] == "body"
    assert data["summary"] == "summary"
    assert data["analysis"]["lang"] == "en"
    assert data["chunks"] == ["chunk"]


def test_read_cli_summarize_non_english(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.fetch_html",
        lambda url: "<html></html>",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.extract_main",
        lambda html, url=None: "cuerpo",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.analyze_text",
        lambda text: {"lang": "es", "word_count": 1},
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.summarize_text",
        lambda text: text,
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.translate_text",
        lambda text, target_lang: "translated",
    )

    result = runner.invoke(
        app,
        ["read", "--url", "http://example.com", "--summarize"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert data["text"] == "cuerpo"
    assert data["summary"] == "translated"


def test_read_cli_translation_missing(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.fetch_html",
        lambda url: "<html></html>",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.extract_main",
        lambda html, url=None: "cuerpo",
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.analyze_text",
        lambda text: {"lang": "es"},
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.news.cli.translate_text",
        lambda text, target_lang: None,
    )

    result = runner.invoke(
        app,
        ["read", "--url", "http://example.com", "--translate", "en"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert "Translation unavailable" in lines[0]
    data = json.loads(lines[-1])
    assert data["text"] == "cuerpo"
    assert "original_text" not in data
