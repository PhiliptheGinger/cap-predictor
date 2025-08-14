import pandas as pd
import numpy as np
from sentimental_cap_predictor.preprocessing import merge_data, preprocess_price_data


def test_merge_data_no_duplicates():
    existing = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "val": [1, 2]})
    new = pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "val": [2, 3]})
    merged = merge_data(existing, new, merge_on="Date")
    assert len(merged) == 3
    assert list(merged["Date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))


def test_preprocess_price_data_scaling():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"close": np.arange(1, 41), "date": dates})
    processed, scaler = preprocess_price_data(df)
    assert processed["close"].min() >= 0
    assert processed["close"].max() <= 1
    # ensure feature engineering added moving average columns
    assert "ma_10" in processed.columns
    assert "ma_30" in processed.columns
