from sentimental_cap_predictor.chatbot_nlu import parse, resolve


def test_daily_pipeline_recognized():
    nlu = parse("please run the daily pipeline", ctx={})
    assert nlu.intent == "pipeline.run_daily"


def test_data_ingest_slots():
    nlu = parse("ingest NVDA for 5d at 1h", ctx={})
    assert nlu.intent == "data.ingest"
    assert nlu.slots["tickers"] == ["NVDA"]
    assert nlu.slots["period"] == "5d"
    assert nlu.slots["interval"] == "1h"


def test_help_intent():
    nlu = parse("help me out", ctx={})
    assert nlu.intent == "help.show_options"
    res = resolve(nlu, ctx={})
    assert res.action_needed == "FALLBACK"


def test_ood_fallback():
    nlu = parse("order pizza", ctx={})
    assert nlu.intent == "help.show_options"
    res = resolve(nlu, ctx={})
    assert res.action_needed == "FALLBACK"
