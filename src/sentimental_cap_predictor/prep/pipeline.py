from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_by_time(df: pd.DataFrame, train_ratio: float = 0.7, gap: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets preserving time order.

    A gap of ``gap`` rows is skipped between the train and test sets to avoid
    lookahead leakage.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx + gap :].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic return features."""
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["log_ret"] = np.log(df["close"]).diff()
    return df


def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators RSI(14), MACD(12,26,9) and ATR(14)."""
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    return df


def validate_no_nans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Raise a ValueError if any of ``cols`` contain NaNs."""
    missing = df[cols].isna().any()
    if missing.any():
        bad_cols = ", ".join(missing[missing].index.tolist())
        raise ValueError(f"NaNs found in columns: {bad_cols}")
    return df
