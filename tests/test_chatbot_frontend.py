# flake8: noqa
import importlib.util
import json
import logging
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests


@dataclass
class ArticleData:
    title: str = ""
    url: str = ""
    content: str = ""


# Provide lightweight stand-ins to avoid importing the full package
dummy_news = types.ModuleType("sentimental_cap_predictor.data.news")
dummy_news.fetch_first_gdelt_article = lambda *a, **k: None


@dataclass
class FetchArticleSpec:
    query: str
    days: int = 1
    max_records: int = 100
    must_contain_any: tuple[str, ...] = ()
    avoid_domains: tuple[str, ...] = ("seekingalpha.com",)
    require_text_accessible: bool = False
    novelty_against_urls: tuple[str, ...] = ()
    language: str | None = "english"


dummy_news.fetch_article = lambda spec, seen_titles=(): ArticleData()
dummy_news.FetchArticleSpec = FetchArticleSpec
dummy_data = types.ModuleType("sentimental_cap_predictor.data")
dummy_data.__path__ = []  # mark as package
dummy_data.news = dummy_news
dummy_pkg = types.ModuleType("sentimental_cap_predictor")
dummy_pkg.__path__ = []
dummy_pkg.data = dummy_data
sys.modules.setdefault("sentimental_cap_predictor", dummy_pkg)
sys.modules.setdefault("sentimental_cap_predictor.data", dummy_data)
sys.modules.setdefault("sentimental_cap_predictor.data.news", dummy_news)
sys.modules.setdefault("sentimental_cap_predictor.data.news", dummy_news)


class _StubMemory:
    def add(self, texts):  # noqa: ANN001
        pass

    def save(self, path):  # noqa: ANN001
        pass

    @classmethod
    def load(cls, path, model_name=None):  # noqa: ANN001
        return cls()


dummy_memory = types.ModuleType("sentimental_cap_predictor.llm_core.memory_indexer")
dummy_memory.TextMemory = _StubMemory
sys.modules.setdefault(
    "sentimental_cap_predictor.llm_core.memory_indexer",
    dummy_memory,
)

dummy_llm_core = types.ModuleType("sentimental_cap_predictor.llm_core")
dummy_llm_core.__path__ = []
dummy_llm_core.memory_indexer = dummy_memory
sys.modules.setdefault("sentimental_cap_predictor.llm_core", dummy_llm_core)

from enum import Enum


class LLMProviderEnum(str, Enum):
    qwen_local = "qwen_local"
    deepseek = "deepseek"


dummy_provider = types.ModuleType(
    "sentimental_cap_predictor.llm_core.provider_config"
)
dummy_provider.LLMProviderEnum = LLMProviderEnum
sys.modules.setdefault(
    "sentimental_cap_predictor.llm_core.provider_config", dummy_provider
)

dummy_engine = types.ModuleType("sentimental_cap_predictor.reasoning.engine")
dummy_engine.analogy_explain = lambda *a, **k: None
dummy_engine.reason_about = lambda *a, **k: None
dummy_engine.simulate = lambda *a, **k: None
sys.modules.setdefault("sentimental_cap_predictor.reasoning.engine", dummy_engine)

# Import the module directly to avoid triggering package-level side effects
spec = importlib.util.spec_from_file_location(
    "chatbot_frontend",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "llm_core"
    / "chatbot_frontend.py",
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


@pytest.fixture(autouse=True)
def _reset_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "_SEEN_META_PATH", tmp_path / "seen.json")
    monkeypatch.setattr(cf, "_MEMORY_INDEX", tmp_path / "memory.faiss")
    monkeypatch.setattr(cf, "_SEEN_METADATA", [])
    monkeypatch.setattr(cf, "_SEEN_URLS", set())
    monkeypatch.setattr(cf, "_SEEN_TITLES", set())
    monkeypatch.setattr(cf, "_SEEN_LOADED", False)
    monkeypatch.setattr(cf, "_LAST_ARTICLE_URL", None)
    monkeypatch.setattr(cf, "_PENDING_TOPIC", False)


def test_fetch_first_gdelt_article(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch(query, *, prefer_content, days=1, max_records=100):  # noqa: ANN001
        captured["prefer_content"] = prefer_content
        return ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        )

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)
    monkeypatch.setattr(cf, "_MEMORY_INDEX", tmp_path / "memory.faiss")

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "Body text"
    assert captured["prefer_content"] is True


def test_seen_sets_update_and_pass(monkeypatch):
    monkeypatch.setattr(cf, "_SEEN_URLS", set())
    monkeypatch.setattr(cf, "_SEEN_TITLES", set())

    captures: list[tuple[str, ...]] = []

    def fake_fetch(spec, seen_titles=()):  # noqa: ANN001
        captures.append(spec.novelty_against_urls)
        return ArticleData(title="Headline", url="http://example.com")

    monkeypatch.setattr(cf, "_fetch_article", fake_fetch)

    cf._fetch_first_gdelt_article("NVDA")
    cf._fetch_first_gdelt_article("NVDA")

    assert captures == [(), ("http://example.com",)]
    assert cf._SEEN_URLS == {"http://example.com"}
    assert cf._SEEN_TITLES == {"Headline"}


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

    from types import SimpleNamespace

    dummy_module = SimpleNamespace(TextMemory=DummyMemory)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.memory_indexer",
        dummy_module,
    )

    monkeypatch.setattr(cf, "_MEMORY_INDEX", index_path)
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(
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

    meta = json.loads(index_path.with_suffix(".json").read_text())
    assert meta == [{"title": "Headline", "url": "http://example.com"}]


def test_fetch_first_gdelt_article_persists_seen(monkeypatch):
    cf._SEEN_META_PATH.write_text('[{"title": "Old", "url": "http://old"}]')
    cf._SEEN_LOADED = False

    def fake_fetch(query, *, prefer_content, days=1, max_records=100):  # noqa: ANN001
        assert "http://old" in cf._SEEN_URLS
        assert "Old" in cf._SEEN_TITLES
        return ArticleData(title="New", url="http://new", content="")

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    cf.fetch_first_gdelt_article("NVDA")

    data = json.loads(cf._SEEN_META_PATH.read_text())
    assert {"title": "Old", "url": "http://old"} in data
    assert {"title": "New", "url": "http://new"} in data


def test_fetch_first_gdelt_article_fallback(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(
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
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(
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
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(
            title="Headline",
            url="http://example.com",
            content="",
        ),
    )

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "Headline - http://example.com"


def test_handle_command_parses_options(monkeypatch):
    captured = {}

    def fake_fetch(query, *, prefer_content, days=1, max_records=100):  # noqa: ANN001
        captured.update(
            query=query,
            prefer_content=prefer_content,
            days=days,
            max_records=max_records,
        )
        return ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        )

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.handle_command('gdelt search --query "climate change" --limit 5 --days 2')
    assert text == "Body text"
    assert captured["query"] == "climate change"
    assert captured["days"] == 2
    assert captured["max_records"] == 5


def test_handle_command_plain_word(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body text",
        ),
    )

    text = cf.handle_command("NVDA")
    assert text == "Body text"


def test_handle_command_fallback(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(),
    )

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "No news found."


def test_fetch_first_gdelt_article_error(monkeypatch):
    def fake_fetch(query, *, prefer_content, days=1, max_records=100):  # noqa: ANN001
        raise requests.RequestException("boom")

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text.startswith("GDELT request failed:")


def test_fetch_first_gdelt_article_runtime_retry(monkeypatch):
    calls: list[bool] = []

    def fake_fetch(query, *, prefer_content, days=1, max_records=100):  # noqa: ANN001
        calls.append(prefer_content)
        if prefer_content:
            raise RuntimeError("boom")
        return ArticleData(title="Headline", url="http://example.com", content="")

    monkeypatch.setattr(cf, "_fetch_first_gdelt_article", fake_fetch)

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "Headline - http://example.com"
    assert cf._LAST_ARTICLE_URL == "http://example.com"
    assert calls == [True, False]


def test_fetch_first_gdelt_article_runtime_error(monkeypatch):
    def fake_fetch(spec, seen_titles=()):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(cf, "_fetch_article", fake_fetch)

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "No readable article found"


def test_handle_command_reports_network_error(monkeypatch):
    def fake_fetch(query, *, prefer_content, days=1, max_records=100):  # noqa: ANN001
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


def test_handle_command_blocks_unapproved_domain(monkeypatch):
    def _fail(*a, **k):  # noqa: ANN001
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(subprocess, "run", _fail)
    out = cf.handle_command("curl https://example.com")
    assert "not permitted" in out.lower()
    assert "<html" not in out.lower()


def test_direct_text_reply(monkeypatch, capsys):
    """The assistant may answer directly without emitting ``CMD:``."""

    responses = ["CMD: echo hi", "You're welcome."]

    class DummyProvider:
        def chat(self, history):  # noqa: ANN001
            return responses.pop(0)

    import sys
    from types import SimpleNamespace

    dummy_module = SimpleNamespace(
        QwenLocalProvider=lambda *a, **k: DummyProvider(),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.llm_providers.qwen_local",
        dummy_module,
    )
    cfg_module = SimpleNamespace(
        QwenLocalConfig=SimpleNamespace(
            from_env=lambda: SimpleNamespace(
                temperature=0.0,
                max_new_tokens=512,
                model_dump=lambda: {"temperature": 0.0, "max_new_tokens": 512},
            )
        )
    )
    llm_core_pkg = SimpleNamespace(provider_config=cfg_module)
    root_pkg = SimpleNamespace(llm_core=llm_core_pkg)
    monkeypatch.setitem(sys.modules, "sentimental_cap_predictor", root_pkg)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core",
        llm_core_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.provider_config",
        cfg_module,
    )
    import importlib.util
    from pathlib import Path

    spec_cmd = importlib.util.spec_from_file_location(
        "sentimental_cap_predictor.cmd_utils",
        Path(__file__).resolve().parents[1]
        / "src"
        / "sentimental_cap_predictor"
        / "cmd_utils.py",
    )
    cmd_utils = importlib.util.module_from_spec(spec_cmd)
    spec_cmd.loader.exec_module(cmd_utils)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.cmd_utils",
        SimpleNamespace(extract_cmd=cmd_utils.extract_cmd),
    )

    user_inputs = iter(["hi", "thanks", "quit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(user_inputs))

    executed = {}

    def fake_handle(cmd):  # noqa: ANN001
        executed["cmd"] = cmd
        return "ok"

    monkeypatch.setattr(cf, "handle_command", fake_handle)
    monkeypatch.setattr(sys, "argv", ["prog"])
    cf.main()

    out = capsys.readouterr().out
    assert executed["cmd"] == "echo hi"
    assert "ok" in out
    assert "You're welcome." in out
    assert "Output invalid" not in out


def test_handle_command_memory_search(monkeypatch, tmp_path):
    index_path = tmp_path / "memory.faiss"
    index_path.touch()
    meta_path = tmp_path / "memory.json"
    meta_path.write_text(
        '[{"title": "First", "url": "http://a"}, '
        '{"title": "Second", "url": "http://b"}]'
    )

    class DummyIndex:
        def search(self, embeddings, k):  # noqa: ANN001
            return [[0.0]], [[0]]

    class DummyMemory:
        def __init__(self):
            self.index = DummyIndex()

        def embed(self, texts):  # noqa: ANN001
            return [[0.0]]

        @classmethod
        def load(cls, path, model_name=None):  # noqa: ANN001
            return cls()

    import sys
    from types import SimpleNamespace

    dummy_module = SimpleNamespace(TextMemory=DummyMemory)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.memory_indexer",
        dummy_module,
    )
    monkeypatch.setattr(cf, "_MEMORY_INDEX", index_path)

    text = cf.handle_command('memory search "query"')
    assert "First - http://a" in text


def test_handle_command_news_fetch_gdelt(monkeypatch):
    import sys
    from types import SimpleNamespace

    captured: dict[str, object] = {}

    def fake_fetch_gdelt_command(query, max_results=3):  # noqa: ANN001
        captured["query"] = query
        captured["max_results"] = max_results
        print("ok")

    dummy_cli = SimpleNamespace(fetch_gdelt_command=fake_fetch_gdelt_command)
    monkeypatch.setitem(sys.modules, "sentimental_cap_predictor.news.cli", dummy_cli)

    out = cf.handle_command("news.fetch_gdelt --query NVDA --max 1")
    assert out == "ok"
    assert captured == {"query": "NVDA", "max_results": 1}


def test_handle_command_news_read(monkeypatch):
    import sys
    from enum import Enum
    from types import SimpleNamespace

    class DummyTranslateMode(str, Enum):
        off = "off"
        en = "en"

    captured: dict[str, object] = {}

    def fake_read_command(
        url,
        *,
        summarize=False,
        analyze=False,
        chunks=None,
        overlap=0,
        translate=DummyTranslateMode.off,
    ):  # noqa: ANN001
        captured.update(
            url=url,
            summarize=summarize,
            analyze=analyze,
            chunks=chunks,
            overlap=overlap,
            translate=translate,
        )
        print("done")

    dummy_cli = SimpleNamespace(
        read_command=fake_read_command, TranslateMode=DummyTranslateMode
    )
    monkeypatch.setitem(sys.modules, "sentimental_cap_predictor.news.cli", dummy_cli)

    out = cf.handle_command(
        "news.read --url http://example.com --summarize --analyze --chunks 1000"
    )
    assert out == "done"
    assert captured["url"] == "http://example.com"
    assert captured["summarize"] is True
    assert captured["analyze"] is True
    assert captured["chunks"] == 1000
    assert captured["overlap"] == 0
    assert captured["translate"] is DummyTranslateMode.off


def test_article_summarize_last(monkeypatch):
    monkeypatch.setattr(
        cf,
        "_fetch_first_gdelt_article",
        lambda query, *, prefer_content, days=1, max_records=100: ArticleData(
            title="Headline",
            url="http://example.com",
            content="Body",
        ),
    )

    cf.handle_command("NVDA")

    captured = {}

    def fake_read_command(
        *, url, summarize=False, analyze=False, chunks=None, overlap=0, translate="off"
    ):
        captured.update(url=url, summarize=summarize)
        print("summary")

    class DummyTranslateMode:
        off = "off"

    from types import SimpleNamespace

    dummy_cli = SimpleNamespace(
        read_command=fake_read_command, TranslateMode=DummyTranslateMode
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.news.cli",
        dummy_cli,
    )

    out = cf.handle_command("article.summarize_last")
    assert out == "summary"
    assert captured["url"] == "http://example.com"
    assert captured["summarize"] is True


def test_article_summarize_last_requires_article():
    assert "No article stored" in cf.handle_command("article.summarize_last")


def test_route_keywords_fetch(monkeypatch):
    captured = {}

    def fake_fetch(topic):  # noqa: ANN001
        captured["topic"] = topic
        cf._LAST_ARTICLE_URL = "http://u"
        return "text"

    monkeypatch.setattr(cf, "fetch_first_gdelt_article", fake_fetch)

    handler = cf._route_keywords("pull up article about NVDA")
    assert handler is not None
    out = handler()
    assert out == "text"
    assert captured["topic"] == "NVDA"
    assert cf._LAST_ARTICLE_URL == "http://u"


def test_route_keywords_fetch_variants(monkeypatch):
    cf._LAST_ARTICLE_URL = None
    captured = {}

    def fake_fetch(topic):  # noqa: ANN001
        captured["topic"] = topic
        cf._LAST_ARTICLE_URL = "http://u"
        return "text"

    monkeypatch.setattr(cf, "fetch_first_gdelt_article", fake_fetch)

    handler = cf._route_keywords("hey, fetch article about NVDA")
    assert handler is not None
    assert handler() == "text"
    assert captured["topic"] == "NVDA"
    assert cf._LAST_ARTICLE_URL == "http://u"

    handler = cf._route_keywords("fetch article about")
    assert handler is not None
    assert handler() == "What topic should I search for?"


def test_route_keywords_fetch_pending_topic(monkeypatch):
    captured = {}

    def fake_fetch(topic):  # noqa: ANN001
        captured["topic"] = topic
        return "text"

    monkeypatch.setattr(cf, "fetch_first_gdelt_article", fake_fetch)

    handler = cf._route_keywords("fetch article about")
    assert handler is not None
    assert handler() == "What topic should I search for?"

    handler = cf._route_keywords("economy")
    assert handler is not None
    assert handler() == "text"
    assert captured["topic"] == "economy"


def test_route_keywords_fetch_with_filler(monkeypatch):
    cf._LAST_ARTICLE_URL = None
    captured = {}

    def fake_fetch(topic):  # noqa: ANN001
        captured["topic"] = topic
        return "text"

    monkeypatch.setattr(cf, "fetch_first_gdelt_article", fake_fetch)

    handler = cf._route_keywords("pull up an article for me about Charlie Kirk")
    assert handler is not None
    assert handler() == "text"
    assert captured["topic"] == "Charlie Kirk"


def test_route_keywords_read_summarize_and_search(monkeypatch):
    cf._LAST_ARTICLE_URL = "http://a"

    captured = []

    def fake_handle(cmd):  # noqa: ANN001
        captured.append(cmd)
        return "ok"

    monkeypatch.setattr(cf, "handle_command", fake_handle)

    handler = cf._route_keywords("read it")
    assert handler is not None
    assert handler() == "ok"
    handler = cf._route_keywords("summarize it")
    assert handler is not None
    assert handler() == "ok"
    handler = cf._route_keywords("hey, search memory for test")
    assert handler is not None
    assert handler() == "ok"

    assert captured == [
        "news.read --url http://a",
        "article.summarize_last",
        'memory search "test"',
    ]


def test_route_keywords_read_summarize_variants(monkeypatch):
    cf._LAST_ARTICLE_URL = "http://a"

    captured = []

    def fake_handle(cmd):  # noqa: ANN001
        captured.append(cmd)
        return "ok"

    monkeypatch.setattr(cf, "handle_command", fake_handle)

    handler = cf._route_keywords("please read that article")
    assert handler is not None
    assert handler() == "ok"
    handler = cf._route_keywords("could you summarize that article?")
    assert handler is not None
    assert handler() == "ok"
    handler = cf._route_keywords("tell me what's on this page")
    assert handler is not None
    assert handler() == "ok"

    assert captured == [
        "news.read --url http://a",
        "article.summarize_last",
        "article.summarize_last",
    ]


def test_route_keywords_last_loaded():
    cf._SEEN_METADATA = [{"title": "T", "url": "http://u"}]
    handler = cf._route_keywords("hey, what did you load?")
    assert handler is not None
    assert handler() == "T - http://u"


def test_route_keywords_reason_simulate_analogy(monkeypatch, caplog):
    monkeypatch.setattr(cf, "reason_about", lambda topic: f"reasoned {topic}")
    handler = cf._route_keywords("hey, reason about inflation")
    assert handler is not None
    with caplog.at_level(logging.INFO):
        assert handler() == "reasoned inflation"

    monkeypatch.setattr(cf, "simulate", lambda scenario: f"sim {scenario}")
    handler = cf._route_keywords("please simulate market crash")
    assert handler is not None
    with caplog.at_level(logging.INFO):
        assert handler() == "sim market crash"

    monkeypatch.setattr(cf, "analogy_explain", lambda src, tgt: f"{src}->{tgt}")
    handler = cf._route_keywords("could you explain with analogy stocks to gambling")
    assert handler is not None
    with caplog.at_level(logging.INFO):
        assert handler() == "stocks->gambling"


def test_classify_and_route_fetch(monkeypatch):
    import re
    import sys
    import types

    dummy_qwen_intent = types.SimpleNamespace(
        call_qwen=lambda _: '<json>{"intent":"info.lookup","slots":{"query":"NVDA"}}</json>',
        _JSON_RE=re.compile(r"<json>\s*(\{.*\})\s*</json>", re.S),
    )
    dummy_pkg = types.SimpleNamespace(qwen_intent=dummy_qwen_intent)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.chatbot_nlu",
        dummy_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.chatbot_nlu.qwen_intent",
        dummy_qwen_intent,
    )
    monkeypatch.setattr(cf, "fetch_first_gdelt_article", lambda q: f"article {q}")

    result = cf._classify_and_route("news about NVDA")
    assert result == "article NVDA"


def test_classify_and_route_unrecognized(monkeypatch):
    import re
    import sys
    import types

    dummy_qwen_intent = types.SimpleNamespace(
        call_qwen=lambda _: '<json>{"intent":"smalltalk.greeting","slots":{}}</json>',
        _JSON_RE=re.compile(r"<json>\s*(\{.*\})\s*</json>", re.S),
    )
    dummy_pkg = types.SimpleNamespace(qwen_intent=dummy_qwen_intent)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.chatbot_nlu",
        dummy_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.chatbot_nlu.qwen_intent",
        dummy_qwen_intent,
    )

    result = cf._classify_and_route("hello there")
    assert result == "Try: pull up an article about X."


def test_classify_and_route_failure(monkeypatch):
    import re
    import sys
    import types

    def boom(_: str) -> str:  # noqa: ANN001
        raise RuntimeError

    dummy_qwen_intent = types.SimpleNamespace(
        call_qwen=boom,
        _JSON_RE=re.compile(r"<json>\s*(\{.*\})\s*</json>", re.S),
    )
    dummy_pkg = types.SimpleNamespace(qwen_intent=dummy_qwen_intent)
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.chatbot_nlu",
        dummy_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_core.chatbot_nlu.qwen_intent",
        dummy_qwen_intent,
    )

    assert cf._classify_and_route("anything") is None
