"""Command line utilities for running the strategy optimizer.

This module provides a small wrapper around :mod:`strategy_optimizer` that
loads processed price data for a given ticker, performs a random search over
moving–average parameters, and saves the best result to JSON.  It is designed to
serve as a convenient command line entry point during experimentation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from . import strategy_optimizer


app = typer.Typer(help="Run random search optimization for a given ticker")


@app.command()
def run(
    ticker: str = typer.Argument(..., help="Ticker symbol (e.g. AAPL)"),
    iterations: int = typer.Option(100, help="Number of random search iterations"),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility"),
    lambda_drawdown: float = typer.Option(
        1.0, help="Penalty for drawdown in walk-forward analysis score"
    ),
) -> None:
    """Optimize strategy parameters for ``ticker`` and save the best result."""

    processed_dir = Path("data/processed")
    csv_path = processed_dir / f"{ticker}_prices.csv"
    if not csv_path.exists():
        raise typer.Exit(code=1)

    logger.info("Loading price data from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    result = strategy_optimizer.random_search(
        df["close"],
        iterations=iterations,
        seed=seed,
        lambda_drawdown=lambda_drawdown,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{ticker}_optimizer_best.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    typer.echo(f"Saved best parameters to {out_path}")


if __name__ == "__main__":
    app()
