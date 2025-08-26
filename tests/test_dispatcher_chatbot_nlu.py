from sentimental_cap_predictor.chatbot_nlu import parse, resolve, dispatch, explain


def test_dispatch_and_argument():
    nlu = parse("run the pipeline now", ctx={})
    res = resolve(nlu, ctx={})
    dec = dispatch(res, ctx={})
    arg = explain(dec, nlu, ctx={})
    assert dec.executed is True
    assert isinstance(arg.text, str) and len(arg.text.split()) >= 8
