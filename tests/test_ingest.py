import pandas as pd
import pytest
import yfinance as yf

from sentimental_cap_predictor.data import ingest


def _sample_download(*args, **kwargs):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, tz="US/Eastern"),
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Adj Close": [1.0, 2.0],
            "Volume": [100, 200],
        }
    )


def test_fetch_prices_schema(monkeypatch):
    monkeypatch.setattr(yf, "download", _sample_download)
    df = ingest.fetch_prices("NVDA")
    assert list(df.columns) == ingest.EXPECTED_COLUMNS
    assert str(df["date"].dt.tz) == "UTC"
    assert df["volume"].dtype == "int64"


def test_fetch_prices_empty(monkeypatch):
    def _empty_download(*args, **kwargs):  # noqa: ANN001
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", _empty_download)
    with pytest.raises(ValueError):
        ingest.fetch_prices("BAD")


def test_save_and_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(yf, "download", _sample_download)
    df = ingest.fetch_prices("NVDA")
    raw_path = ingest.save_prices(df, "NVDA")
    csv_path = ingest.prices_to_csv_for_optimizer(df, "NVDA")
    assert raw_path.exists()
    out_df = pd.read_csv(csv_path)
    assert list(out_df.columns) == ["date", "close"]
