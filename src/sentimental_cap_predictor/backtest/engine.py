"""Backtesting engine with vectorized equity-curve calculation.

This module provides a simple yet efficient backtesting engine that
computes strategy equity curves using vectorised pandas operations. It
supports basic transaction cost modelling via commission and slippage
settings and reports common performance metrics. A Typer CLI is provided
for convenient command line execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import typer


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes
    ----------
    initial_capital: float
        Starting capital for the strategy.
    commission: float
        Commission per trade expressed as a fraction (e.g. ``0.001`` for 0.1%).
    slippage: float
        Slippage per trade expressed as a fraction of traded notional.
    """

    initial_capital: float = 100_000.0
    commission: float = 0.0
    slippage: float = 0.0


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Return the maximum drawdown of an equity curve."""
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1.0
    return float(drawdown.min())


def run_backtest(
    prices: pd.Series, signals: pd.Series, config: BacktestConfig | None = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run a vectorised backtest.

    Parameters
    ----------
    prices: pd.Series
        Asset price series indexed by datetime.
    signals: pd.Series
        Trading signals (1 long, -1 short, 0 flat) aligned with ``prices``.
    config: BacktestConfig, optional
        Backtesting configuration. Defaults to ``BacktestConfig()``.

    Returns
    -------
    result: pd.DataFrame
        DataFrame containing prices, positions, returns and equity curve.
    metrics: Dict[str, float]
        Dictionary of performance metrics.
    """

    config = config or BacktestConfig()

    prices = prices.astype(float)
    signals = signals.reindex(prices.index).fillna(0.0).astype(float)

    returns = prices.pct_change().fillna(0.0)
    positions = signals.shift(1).fillna(0.0)

    trades = positions.diff().abs()
    costs = (config.commission + config.slippage) * trades

    strategy_returns = positions * returns - costs
    equity_curve = (1.0 + strategy_returns).cumprod() * config.initial_capital

    result = pd.DataFrame(
        {
            "price": prices,
            "signal": signals,
            "position": positions,
            "return": strategy_returns,
            "equity_curve": equity_curve,
        }
    )

    total_return = equity_curve.iloc[-1] / config.initial_capital - 1.0
    n_days = len(result)
    cagr = (equity_curve.iloc[-1] / config.initial_capital) ** (252 / n_days) - 1.0
    sharpe = np.nan
    if result["return"].std(ddof=0) > 0:
        sharpe = np.sqrt(252) * result["return"].mean() / result["return"].std(ddof=0)

    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(equity_curve),
    }

    return result, metrics


app = typer.Typer(help="Backtesting utilities")


@app.command()
def run(
    price_csv: Path = typer.Argument(..., help="CSV file containing price series"),
    signal_csv: Path = typer.Argument(..., help="CSV file containing signals"),
    output: Path = typer.Option("equity_curve.csv", help="Where to save equity curve"),
    initial_capital: float = typer.Option(100_000.0, help="Starting capital"),
    commission: float = typer.Option(0.0, help="Commission per trade (fraction)"),
    slippage: float = typer.Option(0.0, help="Slippage per trade (fraction)"),
) -> None:
    """Run a backtest from CSV files and save equity curve."""
    prices = pd.read_csv(price_csv, index_col=0, parse_dates=True).iloc[:, 0]
    signals = pd.read_csv(signal_csv, index_col=0, parse_dates=True).iloc[:, 0]

    result, metrics = run_backtest(
        prices, signals, BacktestConfig(initial_capital, commission, slippage)
    )
    result.to_csv(output)

    for key, value in metrics.items():
        typer.echo(f"{key}: {value:.4f}")


if __name__ == "__main__":
    app()
