import sys
import types
import numpy as np
import pandas as pd

# Inject a lightweight substitute for the TensorFlow-dependent module
build_calls: list[tuple] = []
train_calls: list[bool] = []

dummy_module = types.ModuleType("time_series_deep_learner")


class DummyModel:
    def predict(self, X):
        return np.ones((len(X), 1))


def fake_build_liquid_model(input_shape):
    build_calls.append(input_shape)
    return DummyModel()


def fake_train_model_with_rolling_window(model, X_train, y_train, X_val=None, y_val=None, window_size=100):
    train_calls.append(True)
    return model


dummy_module.build_liquid_model = fake_build_liquid_model
dummy_module.train_model_with_rolling_window = fake_train_model_with_rolling_window

sys.modules["sentimental_cap_predictor.modeling.time_series_deep_learner"] = dummy_module


def test_train_model_sets_seeds_and_trains(monkeypatch):
    np_seeds: list[int] = []
    tf_seeds: list[int] = []

    monkeypatch.setattr(np.random, "seed", lambda s: np_seeds.append(s))
    fake_tf = types.SimpleNamespace(random=types.SimpleNamespace(set_seed=lambda s: tf_seeds.append(s)))
    sys.modules["tensorflow"] = fake_tf

    from sentimental_cap_predictor.model_training import train_model

    price_df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
    model = train_model(price_df, random_state=123)
    assert isinstance(model, DummyModel)
    assert build_calls and train_calls
    assert np_seeds == [123]
    assert tf_seeds == [123]


def test_predict_on_test_data(monkeypatch):
    from sentimental_cap_predictor.model_training import (
        train_model,
        predict_on_test_data,
    )

    price_df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=pd.date_range("2024-01-01", periods=4))
    train_df = price_df.iloc[:2]
    test_df = price_df.iloc[2:]
    sentiment_df = pd.DataFrame(columns=["sentiment"])

    bias_calls: list[bool] = []

    def fake_bias(df, sentiment):
        bias_calls.append(True)
        return df

    monkeypatch.setattr(
        "sentimental_cap_predictor.model_training.bias_predictions_with_sentiment",
        fake_bias,
    )

    model = train_model(train_df)
    result = predict_on_test_data(price_df.copy(), model, test_df.copy(), sentiment_df)
    assert "predicted" in result.columns
    assert bias_calls


def test_predict_on_future_data(monkeypatch):
    from sentimental_cap_predictor.model_training import (
        train_model,
        predict_on_future_data,
    )

    price_df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=pd.date_range("2024-01-01", periods=4))
    sentiment_df = pd.DataFrame(columns=["sentiment"])

    bias_calls: list[bool] = []

    def fake_bias(df, sentiment):
        bias_calls.append(True)
        return df

    monkeypatch.setattr(
        "sentimental_cap_predictor.model_training.bias_predictions_with_sentiment",
        fake_bias,
    )

    model = train_model(price_df)
    result = predict_on_future_data(price_df.copy(), model, 2, sentiment_df)
    assert len(result) == len(price_df) + 2
    assert bias_calls
