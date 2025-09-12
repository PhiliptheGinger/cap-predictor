import importlib
import sys

import pandas as pd

MODULE = "sentimental_cap_predictor.modeling.sentiment_analysis"
AUTO_TOKENIZER = "transformers.AutoTokenizer.from_pretrained"


def test_import_does_not_initialize_pipeline(monkeypatch):
    sys.modules.pop(MODULE, None)

    def fail_pipeline(*args, **kwargs):  # pragma: no cover
        raise AssertionError("pipeline should not be called")

    def fail_tokenizer(*args, **kwargs):  # pragma: no cover
        raise AssertionError("tokenizer should not be called")

    monkeypatch.setattr("transformers.pipeline", fail_pipeline)
    monkeypatch.setattr(AUTO_TOKENIZER, fail_tokenizer)

    importlib.import_module(MODULE)  # should not raise


def test_pipeline_initialized_on_demand(monkeypatch):
    sys.modules.pop(MODULE, None)

    calls = {"pipeline": 0, "tokenizer": 0, "model": None}

    class FakeTokenizer:
        def encode(self, text, truncation=True, max_length=512):
            self.last_text = text
            return [0]

        def decode(self, tokens, skip_special_tokens=True):
            return getattr(self, "last_text", "")

    def fake_pipeline(task, model, framework):
        calls["pipeline"] += 1
        calls["model"] = model

        class _Pipe:
            def __call__(self, text):
                return [{"label": "POSITIVE", "score": 1.0}]

        return _Pipe()

    def fake_from_pretrained(model):
        calls["tokenizer"] += 1
        calls["model"] = model
        return FakeTokenizer()

    monkeypatch.setattr("transformers.pipeline", fake_pipeline)
    monkeypatch.setattr(AUTO_TOKENIZER, fake_from_pretrained)

    sa = importlib.import_module(MODULE)

    df = pd.DataFrame({"content": ["hi"], "date": ["2024-01-01"]})
    result = sa.perform_sentiment_analysis(df, model_name="my-model")

    assert calls["pipeline"] == 1
    assert calls["tokenizer"] == 1
    assert calls["model"] == "my-model"
    assert "weighted_sentiment" in result.columns
