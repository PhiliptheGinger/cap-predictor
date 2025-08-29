"""Utilities for simple strategy optimization.

This module implements a minimal moving–average crossover strategy and a
random search optimizer.  The goal is to provide a lightweight example of a
self‑improving trading component that can be extended in the future.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

app = typer.Typer(help="Utilities for strategy optimization")


@app.callback()
def main() -> None:
    """Entry point for strategy optimizer commands."""
    return None


@dataclass
class OptimizationResult:
    """Container for optimizer results."""

    short_window: int
    long_window: int
    score: float
    mean_return: float
    mean_drawdown: float


def moving_average_crossover(
    prices: pd.Series, short_window: int, long_window: int
) -> float:
    """Evaluate a simple moving-average crossover strategy.

    Parameters
    ----------
    prices:
        Series of closing prices indexed by date.
    short_window, long_window:
        Window sizes for the short and long moving averages.
        ``short_window`` must be strictly less than ``long_window``.

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


def walk_forward_eval(
    prices: pd.Series,
    short_window: int,
    long_window: int,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Evaluate parameters using walk-forward analysis.

    The price series is split into ``n_splits`` sequential folds using
    :class:`~sklearn.model_selection.TimeSeriesSplit`.  For each split the
    strategy is evaluated on the out-of-sample test segment and the total
    return and maximum drawdown are recorded.  The function returns the mean
    return and mean drawdown across all folds.

    Parameters
    ----------
    prices:
        Series of closing prices indexed by date.
    short_window, long_window:
        Window sizes for the short and long moving averages. ``short_window``
        must be strictly less than ``long_window``.
    n_splits:
        Number of walk-forward folds to evaluate.

    Returns
    -------
    tuple[float, float]
        Mean return and mean drawdown across all folds.
    """

    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    returns_list: list[float] = []
    drawdowns: list[float] = []

    for train_idx, test_idx in tscv.split(prices):
        end_idx = test_idx[-1] + 1
        subset = prices.iloc[:end_idx]

        short_ma = subset.rolling(short_window).mean()
        long_ma = subset.rolling(long_window).mean()

        positions = (short_ma > long_ma).astype(int)
        returns = subset.pct_change().fillna(0.0)
        strategy_returns = returns * positions.shift(1).fillna(0.0)

        segment_returns = strategy_returns.iloc[test_idx]
        total_return = float((1 + segment_returns).prod() - 1)
        returns_list.append(total_return)

        equity_curve = (1 + segment_returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = float(drawdown.min())
        drawdowns.append(abs(max_drawdown))

    if returns_list:
        mean_return = float(np.mean(returns_list))
    else:
        mean_return = float("nan")
    if drawdowns:
        mean_drawdown = float(np.mean(drawdowns))
    else:
        mean_drawdown = float("nan")
    return mean_return, mean_drawdown


def random_search(
    prices: pd.Series,
    iterations: int = 100,
    short_range: Tuple[int, int] = (5, 20),
    long_range: Tuple[int, int] = (20, 50),
    seed: int | None = None,
    n_splits: int = 5,
    lambda_drawdown: float = 1.0,
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
        Dataclass containing the best parameters and associated WFA metrics.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if short_range[0] >= short_range[1]:
        raise ValueError("short_range must be an increasing tuple")
    if long_range[0] >= long_range[1]:
        raise ValueError("long_range must be an increasing tuple")

    rng = random.Random(seed)
    best_result: OptimizationResult | None = None

    for _ in tqdm(range(iterations)):
        short = rng.randint(*short_range)
        long = rng.randint(*long_range)
        if short >= long:
            continue

        mean_ret, mean_dd = walk_forward_eval(
            prices,
            short,
            long,
            n_splits=n_splits,
        )
        score = mean_ret - lambda_drawdown * mean_dd
        if not best_result or score > best_result.score:
            best_result = OptimizationResult(
                short,
                long,
                score,
                mean_ret,
                mean_dd,
            )
            logger.debug(
                "New best result: short=%d long=%d score=%.6f return=%.6f "
                "drawdown=%.6f",
                short,
                long,
                score,
                mean_ret,
                mean_dd,
            )

    if best_result is None:
        raise RuntimeError("No valid parameter combinations were evaluated")

    return best_result


@app.command()
def optimize(
    csv_path: str = typer.Argument(
        ..., help="CSV file with 'date' and 'close' columns"
    ),
    iterations: int = typer.Option(
        100,
        help="Number of random search iterations",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed for reproducibility",
    ),
    lambda_drawdown: float = typer.Option(
        1.0,
        help="Penalty for drawdown in WFA score",
    ),
) -> None:
    """Run a random search over moving-average parameters."""

    df = pd.read_csv(csv_path, parse_dates=["date"])
    result = random_search(
        df["close"],
        iterations=iterations,
        seed=seed,
        lambda_drawdown=lambda_drawdown,
    )
    typer.echo(
        f"Best short={result.short_window} long={result.long_window} "
        f"score={result.score:.4f}"
    )


if __name__ == "__main__":
    app()
