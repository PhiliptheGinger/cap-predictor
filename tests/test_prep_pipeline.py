import numpy as np
import pandas as pd
import pytest

from sentimental_cap_predictor.prep import pipeline


def test_train_test_split_by_time_gap():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20),
            "close": np.arange(20, dtype=float),
        }
    )
    train, test = pipeline.train_test_split_by_time(df, train_ratio=0.5, gap=2)
    assert len(train) == 10
    assert len(test) == 8
    assert test["date"].min() > train["date"].max()
    assert (test["date"].min() - train["date"].max()).days == 3


def test_validate_no_nans():
    df = pd.DataFrame(
        {
            "close": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
        }
    )
    df = pipeline.add_returns(df)
    df = pipeline.add_tech_indicators(df)
    with pytest.raises(ValueError):
        pipeline.validate_no_nans(df, ["ret_1d", "rsi_14"])
    df_clean = df.dropna()
    pipeline.validate_no_nans(df_clean, ["ret_1d", "rsi_14"])
