import numpy as np
import pandas as pd

from sentimental_cap_predictor.news.scoring import score_news


def test_score_news_applies_weights_and_ewma():
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates),
            "length": [100, 50, 150],
            "credibility": [0.9, 0.5, 0.1],
        }
    )

    result = score_news(
        df,
        now=pd.Timestamp("2024-01-03"),
        recency_weight=0.5,
        length_weight=0.3,
        credibility_weight=0.2,
        decay=1.0,
        ewma_span=2,
    )

    # Expected components
    now = pd.Timestamp("2024-01-03")
    age_days = (now - df["timestamp"]).dt.total_seconds() / 86400
    recency = np.exp(-age_days)
    length_norm = df["length"] / df["length"].max()
    expected_score = 0.5 * recency
    expected_score += 0.3 * length_norm
    expected_score += 0.2 * df["credibility"]
    expected_score = expected_score.rename("score")
    ewm_series = expected_score.ewm(span=2, adjust=False).mean()
    expected_ewma = ewm_series.rename("ewma_score")

    pd.testing.assert_series_equal(
        result["score"],
        expected_score,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        result["ewma_score"],
        expected_ewma,
        atol=1e-12,
    )
