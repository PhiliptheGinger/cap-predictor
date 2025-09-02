from sentimental_cap_predictor.llm_core import chatbot
from sentimental_cap_predictor.llm_core.chatbot import _predict_intent
from sentimental_cap_predictor.llm_core.chatbot_nlu import qwen_intent


def test_info_lookup_intent_fallback(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("news about NVDA and AAPL")
    assert out["intent"] == "info.lookup"
    assert out["slots"] == {"query": "NVDA and AAPL", "keywords": ["NVDA", "AAPL"]}


def test_info_lookup_dispatch_builds_spec(monkeypatch):
    captured = {}

    def fake_fetch_article(spec):  # noqa: ANN001
        captured["keywords"] = spec.must_contain_any
        from sentimental_cap_predictor.data.news import ArticleData

        return ArticleData(title="Headline", url="http://example.com", content="")

    monkeypatch.setattr(chatbot, "fetch_article", fake_fetch_article)
    out = chatbot.dispatch("info.lookup", {"query": "NVDA", "keywords": ["NVDA"]})
    assert "Headline" in out
    assert captured["keywords"] == ("NVDA",)


def test_info_lookup_dispatch_handles_no_match(monkeypatch):
    def fake_fetch_article(spec):  # noqa: ANN001
        raise RuntimeError("No candidate articles matched filters")

    monkeypatch.setattr(chatbot, "fetch_article", fake_fetch_article)
    msg = chatbot.dispatch("info.lookup", {"query": "NVDA", "keywords": ["NVDA"]})
    assert "couldn't find" in msg.lower()


def test_arxiv_intent_and_dispatch(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("arxiv machine learning")
    assert out == {"intent": "info.arxiv", "slots": {"query": "machine learning"}}

    calls = []

    def fake_fetch(query, max_results=5):
        calls.append((query, max_results))
        return [{"title": "Paper", "updated": "2024-01-01"}]

    monkeypatch.setattr(chatbot.arxiv_connector, "fetch", fake_fetch)
    reply = chatbot.dispatch("info.arxiv", {"query": "ml"})
    assert "Paper" in reply
    assert calls == [("ml", 5)]


def test_pubmed_intent_and_dispatch(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("pubmed cancer research")
    assert out == {"intent": "info.pubmed", "slots": {"query": "cancer research"}}

    calls = []

    def fake_fetch(query, max_results=5):
        calls.append((query, max_results))
        return [{"title": "Study"}]

    monkeypatch.setattr(chatbot.pubmed_connector, "fetch", fake_fetch)
    reply = chatbot.dispatch("info.pubmed", {"query": "heart"})
    assert "Study" in reply
    assert calls == [("heart", 5)]


def test_openalex_intent_and_dispatch(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("openalex reinforcement learning")
    assert out == {"intent": "info.openalex", "slots": {"query": "reinforcement learning"}}

    calls = []

    def fake_fetch(search, per_page=5):
        calls.append((search, per_page))
        return [{"title": "Work"}]

    monkeypatch.setattr(chatbot.openalex_connector, "fetch", fake_fetch)
    reply = chatbot.dispatch("info.openalex", {"query": "rl"})
    assert "Work" in reply
    assert calls == [("rl", 5)]


def test_fred_intent_and_dispatch(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("fred GDP")
    assert out == {"intent": "info.fred", "slots": {"series_id": "GDP"}}

    calls = []

    def fake_fetch_series(series_id, limit=5):
        calls.append((series_id, limit))
        return [{"date": "2024-01-01", "value": "1"}]

    monkeypatch.setattr(chatbot.fred_connector, "fetch_series", fake_fetch_series)
    reply = chatbot.dispatch("info.fred", {"series_id": "GDP"})
    assert "2024-01-01" in reply
    assert calls == [("GDP", 5)]


def test_github_intent_and_dispatch(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("github repo openai/gym")
    assert out == {"intent": "info.github", "slots": {"owner": "openai", "repo": "gym"}}

    calls = []

    def fake_fetch_repo(owner, repo):
        calls.append((owner, repo))
        return {"full_name": f"{owner}/{repo}", "description": "desc"}

    monkeypatch.setattr(chatbot.github_connector, "fetch_repo", fake_fetch_repo)
    reply = chatbot.dispatch("info.github", {"owner": "openai", "repo": "gym"})
    assert "openai/gym" in reply
    assert calls == [("openai", "gym")]
