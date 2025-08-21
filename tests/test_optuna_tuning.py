import sys
import types

import numpy as np
import pandas as pd

# Stub out heavy time_series_deep_learner module
stub_module = types.ModuleType("time_series_deep_learner")


def create_rolling_window_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])  # noqa: E203
        y.append(data[i + window])
    return np.array(X), np.array(y)


class DummyModel:
    def predict(self, X):
        return np.zeros((len(X), 1))


def build_liquid_model(
    input_shape,
    lnn_units=0,
    dropout_rate=0.0,
    learning_rate=0.0,
):
    return DummyModel()


def train_model_with_rolling_window(*args, **kwargs):
    return args[0]


stub_module.create_rolling_window_sequences = create_rolling_window_sequences
stub_module.build_liquid_model = build_liquid_model
stub_module.train_model_with_rolling_window = train_model_with_rolling_window

MODULE_NAME = "sentimental_cap_predictor.modeling.time_series_deep_learner"
sys.modules[MODULE_NAME] = stub_module

from sentimental_cap_predictor.modeling.optuna_tuning import tune  # noqa: E402


def test_tune_updates_env(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=80),
            "Close": np.arange(80),
        }
    )
    csv_path = tmp_path / "data.csv"
    data.to_csv(csv_path, index=False)

    updated = {}

    def fake_set_key(path, key, value):
        updated[key] = value
        return key, value

    monkeypatch.setattr(
        "sentimental_cap_predictor.modeling.optuna_tuning.set_key",
        fake_set_key,
    )

    class DummyStudy:
        def __init__(self):
            self.best_params = {
                "WINDOW_SIZE": 5,
                "LNN_UNITS": 32,
                "DROPOUT_RATE": 0.0,
                "LEARNING_RATE": 1e-5,
                "BATCH_SIZE": 16,
                "EPOCHS": 20,
            }
            self.best_value = 0.0

        def optimize(self, func, n_trials):
            trial = types.SimpleNamespace(
                suggest_int=lambda *a, **k: 5,
                suggest_categorical=lambda *a, **k: a[1][0],
                suggest_float=lambda *a, **k: a[1] if len(a) > 1 else a[0],
            )
            func(trial)

    monkeypatch.setattr(
        "sentimental_cap_predictor.modeling.optuna_tuning.optuna.create_study",
        lambda direction: DummyStudy(),
    )

    tune(data_path=str(csv_path), n_trials=1)

    expected = {
        "WINDOW_SIZE",
        "LNN_UNITS",
        "DROPOUT_RATE",
        "LEARNING_RATE",
        "BATCH_SIZE",
        "EPOCHS",
    }
    assert expected.issubset(updated.keys())
