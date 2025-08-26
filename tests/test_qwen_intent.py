import sentimental_cap_predictor.chatbot_nlu.qwen_intent as qwen_intent


def test_qwen_parses_json_block(monkeypatch):
    engine = qwen_intent.QwenNLU()

    def fake_chat(messages):
        return (
            "<json>{\"intent\":\"data.ingest\",\"slots\":{\"tickers\":[\"nvda\"],\"period\":\"5d\",\"interval\":\"1h\"},\"alt_intent\":\"help.show_options\"}</json>"
        )

    monkeypatch.setattr(engine, "_chat", fake_chat)
    res = engine.predict("ingest NVDA for 5d at 1h")
    assert res.intent == "data.ingest"
    assert res.slots["tickers"] == ["NVDA"]
    assert res.slots["interval"] == "1h"


def test_qwen_fallback_on_parse_error(monkeypatch):
    engine = qwen_intent.QwenNLU()

    def bad_chat(messages):
        return "nonsense"

    monkeypatch.setattr(engine, "_chat", bad_chat)
    res = engine.predict("blah")
    assert res.intent == "help.show_options"
