"""Utilities for loading market, sentiment and fundamental data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from sentimental_cap_predictor.data_bundle import DataBundle


def load_prices(ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Download daily price data for ``ticker`` from Yahoo Finance.

    Columns are converted to lower case for consistency.
    """

    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_sentiment(ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Load daily sentiment scores for ``ticker``.

    Missing scores are forward-filled with ``0.0``.
    """

    idx = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame(index=idx)
    if "score" not in df.columns:
        df["score"] = 0.0
    df["score"] = df["score"].astype(float).fillna(0.0)
    return df


def load_fundamentals(ticker: str) -> pd.DataFrame:
    """Fetch fundamental data for ``ticker``.

    Returns a long-form DataFrame with columns ``date``, ``field``, ``value`` and
    ``asof_ts``.
    """

    ticker_obj = yf.Ticker(ticker)
    frames: list[pd.DataFrame] = []
    for attr in ["balance_sheet", "cashflow", "financials"]:
        data = getattr(ticker_obj, attr, None)
        if data is None or data.empty:
            continue
        for field in data.index:
            series = data.loc[field]
            for date, value in series.items():
                if pd.isna(value):
                    continue
                frames.append(
                    {
                        "date": pd.to_datetime(date),
                        "field": str(field),
                        "value": float(value),
                        "asof_ts": pd.to_datetime(date),
                    }
                )
    if not frames:
        return pd.DataFrame(columns=["date", "field", "value", "asof_ts"])
    return pd.DataFrame(frames)


def align_daily(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align ``df`` to ``index`` using forward fill up to 30 days."""

    return df.reindex(index).ffill(limit=30)


def align_pit_fundamentals(fundamentals: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align point-in-time fundamentals using ``asof_ts`` for forward fill."""

    if fundamentals.empty:
        return pd.DataFrame(index=index)

    out = pd.DataFrame(index=index)
    for field, group in fundamentals.groupby("field"):
        group = group.sort_values("asof_ts")
        series = pd.Series(index=index, dtype=float)
        for _, row in group.iterrows():
            asof = pd.to_datetime(row["asof_ts"])
            series.loc[asof:] = float(row["value"])
        out[field] = series.ffill()
    return out


def make_bundle(ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> DataBundle:
    """Create a :class:`DataBundle` for ``ticker`` between ``start`` and ``end``."""

    prices = load_prices(ticker, start, end)
    sentiment = load_sentiment(ticker, start, end)
    fundamentals_raw = load_fundamentals(ticker)
    sentiment = align_daily(sentiment, prices.index)
    fundamentals = align_pit_fundamentals(fundamentals_raw, prices.index)
    meta = {"ticker": ticker, "start": str(start), "end": str(end)}
    return DataBundle(
        prices=prices,
        features=fundamentals,
        sentiment=sentiment,
        metadata=meta,
    ).validate()

