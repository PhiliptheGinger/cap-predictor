import pandas as pd
from sentimental_cap_predictor.features.builder import build_features


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
