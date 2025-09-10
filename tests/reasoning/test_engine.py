from sentimental_cap_predictor.reasoning import engine


def test_reason_about_no_hits(monkeypatch):
    monkeypatch.setattr(engine.vector_store, "available", lambda: True)
    monkeypatch.setattr(engine.vector_store, "query", lambda text: [])
    result = engine.reason_about("any topic")
    assert "No relevant memories found" in result


def test_analogy_explain_disjoint_dicts():
    src = {"hero": "Alice"}
    tgt = {"villain": "Bob"}
    result = engine.analogy_explain(src, tgt)
    assert "no analogous roles" in result.lower()


def test_simulate_unknown_knobs():
    result = engine.simulate("trend sideways with moderate force")
    assert "generic outcome" in result.lower()
