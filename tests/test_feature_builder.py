import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.features.builder import build_features


FEATURES_PATH = Path(__file__).resolve().parents[1] / "src" / "sentimental_cap_predictor" / "features.py"
spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor._features", FEATURES_PATH
)
features_mod = importlib.util.module_from_spec(spec)
sys.modules["sentimental_cap_predictor._features"] = features_mod
sys.modules.setdefault(
    "sentimental_cap_predictor.model_training",
    types.SimpleNamespace(train_and_predict=lambda *a, **k: None),
)
sys.modules.setdefault(
    "modeling.sentiment_analysis",
    types.SimpleNamespace(perform_sentiment_analysis=lambda df: df),
)
sys.modules.setdefault(
    "sentimental_cap_predictor.dataset",
    types.SimpleNamespace(load_data_bundle=lambda ticker: None),
)
spec.loader.exec_module(features_mod)
del sys.modules["sentimental_cap_predictor.model_training"]
del sys.modules["modeling.sentiment_analysis"]
del sys.modules["sentimental_cap_predictor.dataset"]


def test_build_features_saves_scaler(tmp_path, monkeypatch):
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'open': range(60),
        'high': range(1,61),
        'low': range(60),
        'close': range(60),
        'adj_close': range(60),
        'volume': [100]*60,
    })
    monkeypatch.chdir(tmp_path)
    X, y, d = build_features(df, ticker='TEST')
    assert X.shape[0] == y.shape[0] == len(d)
    scaler_path = tmp_path / 'models' / 'TEST' / 'scaler.pkl'
    assert scaler_path.exists()


def test_calculate_metrics_returns_expected_values():
    scaler = MinMaxScaler().fit([[0], [1]])
    df = pd.DataFrame(
        {
            "close": [0.0, 1.0],
            "predicted": [0.0, 1.0],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )
    df_final, rmse, mape = features_mod.calculate_metrics(df, scaler)
    assert rmse == 0
    assert mape == 0
    assert list(df_final.columns) == ["Date", "actual", "predicted"]


def test_persist_predictions_creates_files(tmp_path):
    df_final = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=2),
            "actual": [1, 2],
            "predicted": [1, 2],
        }
    )
    features_mod.persist_predictions(df_final, "ABC", "train_test", tmp_path, 0.1, 0.2)
    assert (tmp_path / "ABC_train_test_predictions.csv").exists()
    assert (tmp_path / "ABC_train_test_metrics.csv").exists()


def test_generate_predictions_invalid_mode():
    bundle = DataBundle(
        prices=pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")],
            "close": [1.0],
        })
    )
    with pytest.raises(ValueError):
        features_mod.generate_predictions(bundle, "ABC", mode="invalid")
