import types
import sys
import numpy as np
import pandas as pd

# Inject a lightweight substitute for the TensorFlow-dependent module
build_calls = []
train_calls = []

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

from sentimental_cap_predictor.model_training import train_and_predict


def test_train_and_predict_calls_lnn_components(monkeypatch):
    price_df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=pd.date_range("2024-01-01", periods=4))
    train_df = price_df.iloc[:2]
    test_df = price_df.iloc[2:]
    sentiment_df = pd.DataFrame(columns=["date", "sentiment", "confidence"])

    bias_calls = []

    def fake_bias(df, sentiment):
        bias_calls.append(True)
        return df

    monkeypatch.setattr(
        "sentimental_cap_predictor.model_training.bias_predictions_with_sentiment",
        fake_bias,
    )

    result = train_and_predict(
        price_df.copy(),
        train_df.copy(),
        test_df.copy(),
        "train_test",
        prediction_days=2,
        sentiment_df=sentiment_df,
    )

    assert "predicted" in result.columns
    assert build_calls
    assert train_calls
    assert bias_calls

