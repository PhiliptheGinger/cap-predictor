import importlib
import sys
import types

import pandas as pd
import pytest


def test_aggregate_sentiment_by_date_calculates_bias_and_confidence(monkeypatch):
    class DummyTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def encode(self, content, truncation=True, max_length=512):
            return []

        def decode(self, tokens, skip_special_tokens=True):
            return ""

    dummy_transformers = types.SimpleNamespace(
        DistilBertTokenizer=DummyTokenizer,
        pipeline=lambda *args, **kwargs: None,
    )

    sys.modules["transformers"] = dummy_transformers
    sys.modules.pop(
        "sentimental_cap_predictor.modeling.sentiment_analysis", None
    )
    sentiment_analysis = importlib.import_module(
        "sentimental_cap_predictor.modeling.sentiment_analysis"
    )

    df = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ]),
            "weighted_sentiment": [0.5, -0.2, 0.3],
            "confidence": [0.8, 0.2, 0.4],
        }
    )

    aggregated = sentiment_analysis.aggregate_sentiment_by_date(df)

    day1 = aggregated.loc[aggregated["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    day2 = aggregated.loc[aggregated["date"] == pd.Timestamp("2024-01-02")].iloc[0]

    assert day1["bias_factor"] == pytest.approx(0.36)
    assert day1["mean_confidence"] == pytest.approx(0.5)
    assert day2["bias_factor"] == pytest.approx(0.3)
    assert day2["mean_confidence"] == pytest.approx(0.4)
    assert set(aggregated["final_sentiment"]) == {"POSITIVE"}
