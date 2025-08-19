"""Daily pipeline orchestrating data ingestion, preprocessing, model training,
strategy optimization, backtesting and summary reporting.

The pipeline is intentionally lightweight so it can run as a scheduled job.
It persists the final summary to ``data/processed/{ticker}_daily_summary.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from sentimental_cap_predictor.data.ingest import (
    fetch_prices,
    save_prices,
    prices_to_csv_for_optimizer,
)
from sentimental_cap_predictor.preprocessing import preprocess_price_data
from sentimental_cap_predictor.model_training import train_and_predict
from sentimental_cap_predictor.trader_utils.strategy_optimizer import (
    random_search,
    moving_average_crossover,
)

TRAIN_RATIO = 0.8

app = typer.Typer(help="Chain ingestion, preprocessing, modeling and trading evaluation")


def _summary_path(ticker: str) -> Path:
    return Path("data/processed") / f"{ticker}_daily_summary.json"


@app.command()
def run(
    ticker: str = typer.Argument(..., help="Ticker symbol to process"),
    period: str = typer.Option("5y", help="Lookback period for price download"),
    interval: str = typer.Option("1d", help="Price interval"),
) -> None:
    """Execute the end-to-end daily pipeline for ``ticker``."""
    # Ingestion
    prices = fetch_prices(ticker, period=period, interval=interval)
    save_prices(prices, ticker)
    prices_to_csv_for_optimizer(prices, ticker)

    # Preprocessing
    processed, _ = preprocess_price_data(prices)
    split_idx = int(len(processed) * TRAIN_RATIO)
    train_df = processed.iloc[:split_idx]
    test_df = processed.iloc[split_idx:]

    # Model training / prediction
    preds = train_and_predict(
        processed.copy(),
        train_df,
        test_df,
        mode="train_test",
        prediction_days=0,
        sentiment_df=pd.DataFrame(),
    )

    valid = preds.loc[test_df.index].dropna(subset=["predicted"])
    rmse = float(((valid["close"] - valid["predicted"]) ** 2).mean() ** 0.5) if not valid.empty else None

    # Strategy optimization
    opt = random_search(prices["close"])

    # Backtest with best parameters
    backtest_return = moving_average_crossover(
        prices["close"], opt.short_window, opt.long_window
    )

    summary = {
        "ticker": ticker,
        "rmse": rmse,
        "optimizer": {
            "short_window": opt.short_window,
            "long_window": opt.long_window,
            "score": opt.score,
            "mean_return": opt.mean_return,
            "mean_drawdown": opt.mean_drawdown,
            "backtest_return": backtest_return,
        },
    }

    path = _summary_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary report written to %s", path)
    typer.echo(f"Summary report saved to {path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    app()
