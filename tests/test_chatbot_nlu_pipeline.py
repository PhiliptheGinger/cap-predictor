import pytest

from sentimental_cap_predictor.chatbot import _predict_intent
from sentimental_cap_predictor.chatbot_nlu import qwen_intent


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("run the pipeline now", "pipeline.run_now"),
        ("kick off the pipeline right away", "pipeline.run_now"),
        ("please run the daily pipeline", "pipeline.run_daily"),
    ],
)
def test_pipeline_intents_fallback(monkeypatch, phrase, expected):
    # Simulate Qwen failure so the regex fallback triggers
    monkeypatch.setattr(qwen_intent, "predict", lambda _: None)
    out = _predict_intent(phrase)
    assert out["intent"] == expected
