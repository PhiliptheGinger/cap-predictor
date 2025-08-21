"""Build model features and labels with scaling."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
import joblib

from ..prep.pipeline import add_returns, add_tech_indicators


FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "log_ret",
    "rsi_14",
    "macd",
    "macd_signal",
    "atr_14",
]


def build_features(df: pd.DataFrame, ticker: str | None = None):
    """Return standardized feature matrix, labels and dates.

    Parameters
    ----------
    df: pd.DataFrame
        Raw price dataframe containing at least ``date``, ``open``, ``high``,
        ``low`` and ``close`` columns.
    ticker: str, optional
        When provided, the fitted scaler is persisted to
        ``models/{ticker}/scaler.pkl``.
    """

    df = add_returns(df)
    df = add_tech_indicators(df)
    df = df.dropna().reset_index(drop=True)

    # Target is next-day direction
    y = (df["ret_1d"].iloc[1:] > 0).astype(int).reset_index(drop=True)
    df = df.iloc[:-1].copy().reset_index(drop=True)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if ticker:
        model_dir = Path("models") / ticker
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, model_dir / "scaler.pkl")
        logger.debug("Saved scaler to %s", model_dir / "scaler.pkl")

    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    return X_scaled, y.to_numpy(), dates
