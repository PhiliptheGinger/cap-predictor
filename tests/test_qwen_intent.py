from types import SimpleNamespace

import sentimental_cap_predictor.chatbot_nlu.qwen_intent as qwen_intent


def _fake_create(model, temperature, messages):
    utterance = messages[-1]["content"].split("\"")[1]
    mapping = {
        "please run the daily pipeline": "<json>{\"intent\":\"pipeline.run_daily\",\"slots\":{}}</json>",
        "run pipeline": "<json>{\"intent\":\"pipeline.run_now\",\"slots\":{}}</json>",
        "order a pizza": "<json>{\"intent\":\"help.show_options\",\"slots\":{}}</json>",
    }
    content = mapping.get(utterance, "<json>{\"intent\":\"help.show_options\",\"slots\":{}}</json>")
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_smoke_intents(monkeypatch):
    monkeypatch.setattr(qwen_intent.client.chat.completions, "create", _fake_create)

    assert qwen_intent.predict("please run the daily pipeline")["intent"] == "pipeline.run_daily"
    assert qwen_intent.predict("run pipeline")["intent"] == "pipeline.run_now"
    assert qwen_intent.predict("order a pizza")["intent"] == "help.show_options"
