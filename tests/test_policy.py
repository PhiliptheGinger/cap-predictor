from sentimental_cap_predictor.chatbot_nlu import parse, resolve


def test_ambiguous_prompts_trigger_clarify():
    nlu = parse("run the pipeline report", ctx={})
    res = resolve(nlu, ctx={})
    assert res.action_needed == "ASK_CLARIFY"
    assert "pipeline.run_now" in res.prompt and "plots.make_report" in res.prompt


def test_fallback_on_ood():
    nlu = parse("order pizza", ctx={})
    res = resolve(nlu, ctx={})
    assert res.action_needed == "FALLBACK"
