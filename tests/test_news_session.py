import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

# Set up lightweight package structure to avoid heavy imports
root = Path(__file__).resolve().parents[1]
pkg = SimpleNamespace(__path__=[])
sys.modules.setdefault("sentimental_cap_predictor", pkg)

memory_pkg = SimpleNamespace(__path__=[])
sys.modules.setdefault("sentimental_cap_predictor.memory", memory_pkg)
pkg.memory = memory_pkg

news_pkg = SimpleNamespace(__path__=[])
sys.modules.setdefault("sentimental_cap_predictor.news", news_pkg)
pkg.news = news_pkg

# Load actual session_state module (it's lightweight)
ss_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.memory.session_state",
    root / "src" / "sentimental_cap_predictor" / "memory" / "session_state.py",
)
session_state_mod = importlib.util.module_from_spec(ss_spec)
name = "sentimental_cap_predictor.memory.session_state"
sys.modules[name] = session_state_mod
ss_spec.loader.exec_module(session_state_mod)
memory_pkg.session_state = session_state_mod

# Provide stub vector_store module
vector_store_mod = SimpleNamespace(
    query=lambda q: [],
    upsert=lambda *a, **k: None,
)
sys.modules["sentimental_cap_predictor.memory.vector_store"] = vector_store_mod
memory_pkg.vector_store = vector_store_mod

# Stub fetch_gdelt module for the session helpers
fetch_gdelt_mod = SimpleNamespace(
    search_gdelt=lambda *a, **k: [],
    fetch_html=lambda url: "",
    extract_main_text=lambda html, url=None: "",
    summarize=lambda text: "",
    score_sentiment=lambda text: 0.0,
    score_relevance=lambda text, query: 0.0,
    domain_ok=lambda url: True,
    domain_blocked=lambda url: False,
    _store_chunks=lambda result: None,
    _chunk_text=lambda text: [text],
    _is_empty_page=lambda html: False,
)
sys.modules["sentimental_cap_predictor.news.fetch_gdelt"] = fetch_gdelt_mod
news_pkg.fetch_gdelt = fetch_gdelt_mod

# Now load the session module under test with package name for relative imports
spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.session",
    root / "src" / "sentimental_cap_predictor" / "news" / "session.py",
)
session = importlib.util.module_from_spec(spec)
sys.modules["sentimental_cap_predictor.news.session"] = session
spec.loader.exec_module(session)


def test_handle_fetch_sets_state_and_upserts(monkeypatch):
    calls = []

    def fake_search_gdelt(query, max_records=15):  # noqa: ANN001
        return [
            {
                "title": f"Title {query}",
                "url": f"http://{query}.com",
                "domain": "example.com",
                "language": "en",
                "seendate": "today",
            }
        ]

    def fake_fetch_html(url):  # noqa: ANN001
        return "<html><body>content</body></html>"

    def fake_extract(html, url=None):  # noqa: ANN001
        return "Body text"

    def fake_summarize(text):  # noqa: ANN001
        return "Summary"

    def fake_sentiment(text):  # noqa: ANN001
        return 0.5

    def fake_relevance(text, topic):  # noqa: ANN001
        return 0.8

    def fake_upsert(doc_id, text, metadata):  # noqa: ANN001
        calls.append((doc_id, text, metadata))

    monkeypatch.setattr(
        session,
        "vector_store",
        SimpleNamespace(query=lambda q: [], upsert=fake_upsert),
    )
    monkeypatch.setattr(session, "STATE", session.STATE.__class__())

    fg_mod = sys.modules["sentimental_cap_predictor.news.fetch_gdelt"]
    monkeypatch.setattr(fg_mod, "search_gdelt", fake_search_gdelt)
    monkeypatch.setattr(fg_mod, "fetch_html", fake_fetch_html)
    monkeypatch.setattr(fg_mod, "extract_main_text", fake_extract)
    monkeypatch.setattr(fg_mod, "summarize", fake_summarize)
    monkeypatch.setattr(fg_mod, "score_sentiment", fake_sentiment)
    monkeypatch.setattr(fg_mod, "score_relevance", fake_relevance)
    monkeypatch.setattr(fg_mod, "domain_ok", lambda url: True)
    monkeypatch.setattr(fg_mod, "domain_blocked", lambda url: False)
    monkeypatch.setattr(
        fg_mod,
        "_store_chunks",
        lambda result: fake_upsert("id", result["text"], {}),
    )
    monkeypatch.setattr(fg_mod, "_chunk_text", lambda text: [text])

    msg = session.handle_fetch("topic1")
    assert (
        msg
        == "Loaded: Title topic1 — http://topic1.com. Say \"read it\" or \"summarize it\"."
    )
    assert session.STATE.last_article["title"] == "Title topic1"
    assert calls  # upsert called
    assert session.STATE.last_article["sentiment"] == 0.5
    assert session.STATE.last_article["relevance"] == 0.8

    # Fetch another topic to ensure state replacement
    msg2 = session.handle_fetch("topic2")
    assert session.STATE.last_article["title"] == "Title topic2"
    assert (
        msg2
        == "Loaded: Title topic2 — http://topic2.com. Say \"read it\" or \"summarize it\"."
    )


def test_handle_fetch_unreadable_page(monkeypatch):
    monkeypatch.setattr(session, "STATE", session.STATE.__class__())

    def fake_search_gdelt(query, max_records=15):  # noqa: ANN001
        return [
            {
                "title": "T",
                "url": "http://t.com",
                "domain": "t.com",
                "language": "en",
                "seendate": "today",
            }
        ]

    fg_mod = sys.modules["sentimental_cap_predictor.news.fetch_gdelt"]
    monkeypatch.setattr(fg_mod, "search_gdelt", fake_search_gdelt)
    monkeypatch.setattr(
        fg_mod, "fetch_html", lambda url: "<html><head></head><body></body></html>"
    )
    monkeypatch.setattr(fg_mod, "domain_ok", lambda url: True)
    monkeypatch.setattr(fg_mod, "domain_blocked", lambda url: False)

    msg = session.handle_fetch("topic")
    assert msg == "Couldn't fetch a readable article; try another topic."


def test_handle_read_and_summarize(monkeypatch):
    monkeypatch.setattr(session, "STATE", session.STATE.__class__())

    # No article loaded
    assert "No article" in session.handle_read()
    assert "No article" in session.handle_summarize()

    # Load article
    session.STATE.set_article({"text": "x" * 2500})
    # Read should truncate
    assert len(session.handle_read()) == 2000

    # Summarize with existing summary
    session.STATE.last_article["summary"] = "sum"
    assert session.handle_summarize() == "sum"

    # Remove summary, expect generation
    session.STATE.last_article.pop("summary")
    fg_mod = sys.modules["sentimental_cap_predictor.news.fetch_gdelt"]
    monkeypatch.setattr(fg_mod, "summarize", lambda text: "generated")
    assert session.handle_summarize() == "generated"
    assert session.STATE.last_article["summary"] == "generated"


def test_handle_memory_search(monkeypatch):
    results = [
        {"metadata": {"title": "A", "url": "http://a"}},
        {"metadata": {"title": "B", "url": "http://b"}},
    ]

    monkeypatch.setattr(
        session,
        "vector_store",
        SimpleNamespace(query=lambda q: results, available=lambda: True),
    )
    text = session.handle_memory_search("q")
    assert "A — http://a" in text
    assert "B — http://b" in text
    monkeypatch.setattr(
        session,
        "vector_store",
        SimpleNamespace(query=lambda q: [], available=lambda: True),
    )
    assert session.handle_memory_search("q") == "No matches found."


def test_handle_memory_search_fallback(monkeypatch):
    monkeypatch.setattr(session, "STATE", session.STATE.__class__())
    session.STATE.recent_chunks = ["alpha beta", "gamma delta"]
    monkeypatch.setattr(
        session,
        "vector_store",
        SimpleNamespace(query=lambda q: [], available=lambda: False),
    )
    assert "alpha beta" in session.handle_memory_search("alpha")
