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

    Signals are interpreted as target weights for the first asset in
    ``data.prices``.  Trades are executed at the same bar with optional fixed
    transaction cost and proportional slippage.
    """

    prices = data.prices.iloc[:, 0]
    weights = strategy.generate_signals(data).reindex(prices.index).fillna(0)
    weight_series = weights.iloc[:, 0]

    position = 0.0
    cash = initial_capital
    equity_curve = []
    trades = []
    trade_pnls = []
    holding_periods = []
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None

    for date, price in prices.items():
        desired_weight = weight_series.loc[date]
        equity = cash + position * price
        desired = desired_weight * equity / price
        if desired != position:
            trade_size = desired - position
            trade_price = price * (1 + slippage if trade_size > 0 else 1 - slippage)
            cash -= trade_size * trade_price
            cash -= cost_per_trade

            if position != 0:
                pnl = (trade_price - entry_price) * position  # type: ignore
                trade_pnls.append(pnl)
                holding = (date - entry_date).days if isinstance(date, pd.Timestamp) else date - entry_date  # type: ignore
                holding_periods.append(holding)
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "size": position,
                        "entry_price": entry_price,
                        "exit_price": trade_price,
                        "pnl": pnl,
                        "holding_period": holding,
                    }
                )

            if desired != 0:
                entry_price = trade_price
                entry_date = date
            else:
                entry_price = None
                entry_date = None

            position = desired

        equity_curve.append({"date": date, "equity": cash + position * price})

    equity_series = pd.Series([e["equity"] for e in equity_curve], index=prices.index)
    trades_df = pd.DataFrame(trades)
    trade_pnls_series = pd.Series(trade_pnls)
    holding_series = pd.Series(holding_periods)
    metrics = compute_metrics(equity_series, trades_df)
    parameters = {
        "initial_capital": initial_capital,
        "cost_per_trade": cost_per_trade,
        "slippage": slippage,
        "strategy": strategy.__class__.__name__,
    }
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity_series,
        metrics=metrics,
        parameters=parameters,
        trade_pnls=trade_pnls_series,
        holding_periods=holding_series,
    )
