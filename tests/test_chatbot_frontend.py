import subprocess
import requests
import importlib.util
from pathlib import Path
from dataclasses import dataclass
import sys
import types


@dataclass
class ArticleData:
    title: str = ""
    url: str = ""
    content: str = ""


# Provide lightweight stand-ins to avoid importing the full package
dummy_news = types.ModuleType("sentimental_cap_predictor.data.news")
dummy_news.fetch_first_gdelt_article = lambda *a, **k: None
dummy_data = types.ModuleType("sentimental_cap_predictor.data")
dummy_data.__path__ = []  # mark as package
dummy_data.news = dummy_news
sys.modules.setdefault("sentimental_cap_predictor.data", dummy_data)
sys.modules.setdefault("sentimental_cap_predictor.data.news", dummy_news)


class _StubMemory:
    def add(self, texts):  # noqa: ANN001
        pass

    def save(self, path):  # noqa: ANN001
        pass

    @classmethod
    def load(cls, path, model_name=None):  # noqa: ANN001
        return cls()


dummy_memory = types.ModuleType("sentimental_cap_predictor.memory_indexer")
dummy_memory.TextMemory = _StubMemory
sys.modules.setdefault("sentimental_cap_predictor.memory_indexer", dummy_memory)

# Import the module directly to avoid triggering package-level side effects
spec = importlib.util.spec_from_file_location(
    "chatbot_frontend",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "chatbot_frontend.py",
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


def test_fetch_first_gdelt_article(monkeypatch):
    captured = {}

    def fake_fetch(query, *, prefer_content):  # noqa: ANN001
        captured["prefer_content"] = prefer_content
        return ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        )

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "Body text"
    assert captured["prefer_content"] is True


def test_fetch_first_gdelt_article_appends_memory(monkeypatch, tmp_path):
    index_path = tmp_path / "memory.faiss"
    index_path.touch()

    class DummyMemory:
        added: list[str] = []
        saved_path = None
        loaded = False

        def add(self, texts):  # noqa: ANN001
            DummyMemory.added.extend(texts)

        def save(self, path):  # noqa: ANN001
            DummyMemory.saved_path = path

        @classmethod
        def load(cls, path, model_name=None):  # noqa: ANN001
            cls.loaded = True
            return cls()

    import sys
    from types import SimpleNamespace

    dummy_module = SimpleNamespace(TextMemory=DummyMemory)
    monkeypatch.setitem(sys.modules, "sentimental_cap_predictor.memory_indexer", dummy_module)

    monkeypatch.setattr(cf, "_MEMORY_INDEX", index_path)
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content: ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        ),
    )

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "Body text"
    assert DummyMemory.added == ["Body text"]
    assert DummyMemory.saved_path == index_path
    assert DummyMemory.loaded is True


def test_fetch_first_gdelt_article_fallback(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content: ArticleData(
            title="Headline",
            url="http://example.com",
            content="",
        ),
    )

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "Headline - http://example.com"


def test_handle_command_routes_to_gdelt(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content: ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        ),
    )

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "Body text"


def test_handle_command_returns_headline(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content: ArticleData(
            title="Headline",
            url="http://example.com",
            content="",
        ),
    )

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "Headline - http://example.com"


def test_handle_command_parses_dash_query(monkeypatch):
    captured = {}

    def fake_fetch(query, *, prefer_content):  # noqa: ANN001
        captured["query"] = query
        return ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        )

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.handle_command(
        'gdelt search --query "climate change" --limit 5'
    )
    assert text == "Body text"
    assert captured["query"] == "climate change"


def test_handle_command_fallback(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content: ArticleData(),
    )

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "No news found."


def test_fetch_first_gdelt_article_error(monkeypatch):
    def fake_fetch(query, *, prefer_content):  # noqa: ANN001
        raise requests.RequestException("boom")

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text.startswith("GDELT request failed:")


def test_handle_command_reports_network_error(monkeypatch):
    def fake_fetch(query, *, prefer_content):  # noqa: ANN001
        raise requests.RequestException("boom")

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text.startswith("GDELT request failed:")


def test_handle_command_runs_shell(monkeypatch):
    class DummyCompleted:
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: DummyCompleted())
    out = cf.handle_command("echo hi")
    assert out == "ok"


def test_retry_on_malformed_output(monkeypatch):
    """Ensure malformed replies trigger a single retry with a reminder."""

    outputs = ["unexpected", "CMD: echo hi"]
    calls: list[str] = []

    class DummyProvider:
        def chat(self, history):  # noqa: ANN001
            calls.append(history[-1]["content"])
            return outputs.pop(0)

    import sys
    from types import SimpleNamespace

    dummy_module = SimpleNamespace(
        QwenLocalProvider=lambda *a, **k: DummyProvider(),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_providers.qwen_local",
        dummy_module,
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.config_llm.get_llm_config",
        lambda: SimpleNamespace(model_path="", temperature=0.0),
    )

    user_inputs = iter(["hi", "quit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(user_inputs))

    executed = {}

    def fake_handle(cmd):  # noqa: ANN001
        executed["cmd"] = cmd
        return "done"

    monkeypatch.setattr(cf, "handle_command", fake_handle)
    cf.main()

    assert calls == [
        "hi",
        "Output invalid. Remember the CMD contract.",
    ]
    assert executed["cmd"] == "echo hi"
