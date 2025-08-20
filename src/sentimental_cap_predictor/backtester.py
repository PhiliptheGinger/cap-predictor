from __future__ import annotations

from typing import Optional

import pandas as pd

from .data_bundle import DataBundle
from .strategy import StrategyIdea
from .backtest_result import BacktestResult
from .evaluation import compute_metrics


def backtest(
    data: DataBundle,
    strategy: StrategyIdea,
    initial_capital: float = 1_000_000.0,
    cost_per_trade: float = 0.0,
    slippage: float = 0.0,
) -> BacktestResult:
    """Run a minimal backtest for ``strategy`` using ``data``.

    Signals are interpreted as target position (number of units) in the first
    column of ``data.prices``.  Trades are executed at the same bar with optional
    fixed transaction cost and proportional slippage.
    """

    prices = data.prices.iloc[:, 0]
    signals = strategy.generate_signals(data).reindex(prices.index).fillna(0)

    position = 0.0
    cash = initial_capital
    equity_curve = []
    trades = []

    for date, price in prices.items():
        desired = signals.loc[date]
        if desired != position:
            trade_size = desired - position
            trade_price = price * (1 + slippage if trade_size > 0 else 1 - slippage)
            cash -= trade_size * trade_price
            cash -= cost_per_trade
            trades.append({"date": date, "size": trade_size, "price": trade_price})
            position = desired
        equity_curve.append({"date": date, "equity": cash + position * price})

    equity_series = pd.Series([e["equity"] for e in equity_curve], index=prices.index)
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(equity_series, trades_df)
    return BacktestResult(trades=trades_df, equity_curve=equity_series, metrics=metrics)
