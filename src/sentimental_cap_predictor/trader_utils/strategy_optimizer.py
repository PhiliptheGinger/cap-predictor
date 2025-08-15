"""Utilities for simple strategy optimization.

This module implements a minimal moving–average crossover strategy and a
random search optimizer.  The goal is to provide a lightweight example of a
self‑improving trading component that can be extended in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import random

import numpy as np
import pandas as pd
import typer
from loguru import logger


app = typer.Typer()


@dataclass
class OptimizationResult:
    """Container for optimizer results."""

    short_window: int
    long_window: int
    total_return: float


def moving_average_crossover(prices: pd.Series, short_window: int, long_window: int) -> float:
    """Evaluate a simple moving-average crossover strategy.

    Parameters
    ----------
    prices:
        Series of closing prices indexed by date.
    short_window, long_window:
        Window sizes for the short and long moving averages. ``short_window`` must
        be strictly less than ``long_window``.

    Returns
    -------
    float
        Total return of the strategy over the period.
    """

    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    short_ma = prices.rolling(short_window).mean()
    long_ma = prices.rolling(long_window).mean()

    positions = (short_ma > long_ma).astype(int)
    returns = prices.pct_change().fillna(0.0)
    strategy_returns = returns * positions.shift(1).fillna(0.0)
    total_return = float((1 + strategy_returns).prod() - 1)
    return total_return


def random_search(
    prices: pd.Series,
    iterations: int = 100,
    short_range: Tuple[int, int] = (5, 20),
    long_range: Tuple[int, int] = (20, 50),
    seed: int | None = None,
) -> OptimizationResult:
    """Randomly search for profitable moving-average parameters.

    Parameters
    ----------
    prices:
        Series of closing prices indexed by date.
    iterations:
        Number of random parameter sets to evaluate.
    short_range, long_range:
        Inclusive ranges from which to sample window sizes.
    seed:
        Optional seed for reproducible results.

    Returns
    -------
    OptimizationResult
        Dataclass containing the best parameters and associated return.
    """

    rng = random.Random(seed)
    best_result: OptimizationResult | None = None

    for _ in range(iterations):
        short = rng.randint(*short_range)
        long = rng.randint(*long_range)
        if short >= long:
            continue

        performance = moving_average_crossover(prices, short, long)
        if not best_result or performance > best_result.total_return:
            best_result = OptimizationResult(short, long, performance)
            logger.debug(
                "New best result: short=%d long=%d return=%.6f",
                short,
                long,
                performance,
            )

    if best_result is None:
        raise RuntimeError("No valid parameter combinations were evaluated")

    return best_result


@app.command()
def optimize(
    csv_path: str = typer.Argument(..., help="CSV file with 'date' and 'close' columns"),
    iterations: int = typer.Option(100, help="Number of random search iterations"),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility"),
) -> None:
    """Run a random search over moving-average parameters."""

    df = pd.read_csv(csv_path, parse_dates=["date"])
    result = random_search(df["close"], iterations=iterations, seed=seed)
    typer.echo(
        f"Best short={result.short_window} long={result.long_window} return={result.total_return:.4f}"
    )


if __name__ == "__main__":
    app()
