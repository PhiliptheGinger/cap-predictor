"""Price data ingestion and normalization utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yfinance as yf
from loguru import logger

EXPECTED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def fetch_prices(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch price data for *ticker* using yfinance.

    The returned dataframe has lower-case columns and UTC-normalized ``date``.
    Retries up to three times with exponential backoff if no data is returned.
    """

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            logger.debug("Downloading prices for %s (attempt %d)", ticker, attempt + 1)
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2**attempt)
    else:
        raise ValueError(f"No data returned for ticker {ticker}") from last_error

    df = df.reset_index().rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset="date").sort_values("date")
    df = df[EXPECTED_COLUMNS]
    df = df.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "adj_close": "float64",
            "volume": "int64",
        }
    )
    return df


def save_prices(df: pd.DataFrame, ticker: str) -> Path:
    """Save raw prices to ``data/raw/{ticker}_prices.parquet``."""

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{ticker}_prices.parquet"
    df.to_parquet(path, index=False)
    return path


def prices_to_csv_for_optimizer(df: pd.DataFrame, ticker: str) -> Path:
    """Write close prices to ``data/processed/{ticker}_prices.csv``."""

    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    path = proc_dir / f"{ticker}_prices.csv"
    df[["date", "close"]].to_csv(path, index=False)
    return path


app = typer.Typer(help="Download and prepare price data")


@app.command()
def main(ticker: str, period: str = "5y", interval: str = "1d") -> None:
    """Fetch prices and materialize parquet/CSV files."""

    df = fetch_prices(ticker, period=period, interval=interval)
    save_path = save_prices(df, ticker)
    csv_path = prices_to_csv_for_optimizer(df, ticker)
    typer.echo(f"Saved parquet to {save_path} and csv to {csv_path}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    app()
