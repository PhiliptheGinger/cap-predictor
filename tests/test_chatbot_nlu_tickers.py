import pytest

from sentimental_cap_predictor.chatbot_nlu import qwen_intent


@pytest.mark.parametrize(
    "phrase,intent,slots",
    [
        ("ingest NVDA AAPL", "data.ingest", {"tickers": ["NVDA", "AAPL"]}),
        ("train NVDA", "model.train_eval", {"ticker": "NVDA"}),
        ("plot TSLA", "plots.make_report", {"ticker": "TSLA"}),
    ],
)
def test_predict_fallback_ticker_slots(phrase, intent, slots):
    out = qwen_intent.predict_fallback(phrase)
    assert out["intent"] == intent
    assert out["slots"] == slots
