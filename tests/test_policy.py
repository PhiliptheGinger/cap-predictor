from sentimental_cap_predictor.chatbot_nlu import parse, resolve


def test_resolve_dispatches_known_intent():
    nlu = parse("run the pipeline now", ctx={})
    res = resolve(nlu, ctx={})
    assert res.action_needed == "DISPATCH"
    assert res.intent == "pipeline.run_now"


def test_fallback_on_ood():
    nlu = parse("order pizza", ctx={})
    res = resolve(nlu, ctx={})
    assert res.action_needed == "FALLBACK"
    assert res.intent == "help.show_options"

