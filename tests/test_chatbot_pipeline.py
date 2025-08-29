from sentimental_cap_predictor import chatbot


def test_pipeline_intents_call_helper(monkeypatch):
    calls = []

    def fake_helper(slots):
        calls.append(slots)
        return "ok"

    monkeypatch.setattr(chatbot, "_run_pipeline_from_slots", fake_helper)

    slots = {"ticker": "TSLA", "period": "1y", "interval": "1d"}

    out = chatbot.dispatch("pipeline.run_now", slots)
    assert out == "ok"

    out = chatbot.dispatch("pipeline.run_daily", slots)
    assert out == "ok"

    assert calls == [slots, slots]
