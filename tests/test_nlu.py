from sentimental_cap_predictor.chatbot_nlu import parse


def test_daily_pipeline_recognized():
    nlu = parse("please run the daily pipeline", ctx={})
    assert nlu.intent == "pipeline.run_daily"
    assert nlu.scores["pipeline.run_daily"] >= 0.72
    assert max(v for k, v in nlu.scores.items() if k != "pipeline.run_daily") < 0.64


def test_data_ingest_slots():
    nlu = parse("ingest NVDA and AAPL for 5d at 1h", ctx={})
    assert nlu.intent == "data.ingest"
    assert set(nlu.slots["tickers"]) == {"NVDA", "AAPL"}
    assert nlu.slots["period"] == "5d"
    assert nlu.slots["interval"] == "1h"
