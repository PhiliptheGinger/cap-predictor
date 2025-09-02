"""Moving average crossover strategy.

This module generates simple moving average (MA) crossover trading
signals. It exposes a Typer CLI for creating a signal file that can be
consumed by the backtesting engine.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer


def generate_signals(
    prices: pd.Series, short_window: int = 20, long_window: int = 50
) -> pd.Series:
    """Create MA crossover signals.

    A signal of ``1`` indicates a long position when the short moving
    average is above the long moving average, otherwise ``0``.
    """

    short_ma = prices.rolling(window=short_window, min_periods=1).mean()
    long_ma = prices.rolling(window=long_window, min_periods=1).mean()
    signals = (short_ma > long_ma).astype(int)
    return signals


app = typer.Typer(help="Generate moving average crossover signals")


@app.command()
def save(
    price_csv: Path = typer.Argument(..., help="CSV file containing price series"),
    output: Path = typer.Option("signals.csv", help="Where to save signals"),
    short_window: int = typer.Option(20, help="Short moving average window"),
    long_window: int = typer.Option(50, help="Long moving average window"),
) -> None:
    """Generate MA crossover signals from prices and save to CSV."""
    prices = pd.read_csv(price_csv, index_col=0, parse_dates=True).iloc[:, 0]
    signals = generate_signals(prices, short_window, long_window)
    pd.DataFrame({"signal": signals}).to_csv(output)


if __name__ == "__main__":
    app()
