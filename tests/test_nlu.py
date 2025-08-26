from sentimental_cap_predictor.chatbot_nlu import parse, resolve


def test_daily_pipeline_recognized():
    nlu = parse("please run the daily pipeline", ctx={})
    assert nlu.intent == "pipeline.run_daily"


def test_data_ingest_slots():
    nlu = parse("ingest NVDA and AAPL for 5d at 1h", ctx={})
    assert nlu.intent == "data.ingest"
    assert set(nlu.slots["tickers"]) == {"NVDA", "AAPL"}
    assert nlu.slots["period"] == "5d"
    assert nlu.slots["interval"] == "1h"


def test_help_intent():
    nlu = parse("help me out", ctx={})
    assert nlu.intent == "help.show_options"


def test_order_pizza_fallback():
    nlu = parse("order pizza", ctx={})
    res = resolve(nlu, ctx={})
    assert res.intent == "help.show_options"
    assert res.action_needed == "FALLBACK"

