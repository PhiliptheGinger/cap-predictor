import pandas as pd
import pytest

from sentimental_cap_predictor import chatbot
from sentimental_cap_predictor.chatbot import _predict_intent
from sentimental_cap_predictor.chatbot_nlu import qwen_intent


def test_info_lookup_intent_fallback(monkeypatch):
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent("news about NVDA")
    assert out["intent"] == "info.lookup"
    assert out["slots"] == {"query": "NVDA"}


def test_info_lookup_dispatch_calls_fetch_news(monkeypatch):
    calls = []

    def fake_fetch_news(query):
        calls.append(query)
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "headline": ["Sample headline"],
                "source": ["Test"],
            }
        )

    monkeypatch.setattr(chatbot, "fetch_news", fake_fetch_news)
    out = chatbot.dispatch("info.lookup", {"query": "NVDA"})
    assert "Sample headline" in out
    assert calls == ["NVDA"]
