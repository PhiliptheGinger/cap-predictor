from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from ..data_bundle import DataBundle
from .strategy import StrategyIdea
from .backtest_result import BacktestResult
from ..evaluation import compute_metrics


def backtest(
    data: DataBundle,
    strategy: StrategyIdea,
    initial_capital: float = 1_000_000.0,
    cost_per_trade: float | Mapping[str, float] = 0.0,
    slippage: float | Mapping[str, float] = 0.0,
) -> BacktestResult:
    """Run a minimal multi-asset backtest for ``strategy`` using ``data``.

    Signals are interpreted as target weights for all assets contained in
    ``data.prices``.  Orders derived from today's signals are executed on the
    next bar, allowing both long and short positions.  Transaction costs and
    slippage may be specified per asset via mappings.  Fill prices including
    slippage are recorded in the trade log.
    """

    # ------------------------------------------------------------------
    # Prepare price data and signals
    # ------------------------------------------------------------------
    if isinstance(data.prices.columns, pd.MultiIndex):
        prices = data.prices.xs("close", level=-1, axis=1)
    else:
        prices = data.prices

    weights = strategy.generate_signals(data)
    weights = weights.reindex(prices.index).reindex(columns=prices.columns, fill_value=0.0)
    # Execute on next bar
    weights = weights.shift(1).fillna(0.0)

    assets = list(prices.columns)
    get_cost = (lambda a: cost_per_trade.get(a, 0.0)) if isinstance(cost_per_trade, Mapping) else (lambda a: cost_per_trade)
    get_slip = (lambda a: slippage.get(a, 0.0)) if isinstance(slippage, Mapping) else (lambda a: slippage)

    positions = pd.Series(0.0, index=assets)
    cash = initial_capital

    equity_curve = []
    trades = []
    trade_pnls: list[float] = []
    holding_periods: list[int | float] = []
    entry_price: dict[str, Optional[float]] = {a: None for a in assets}
    entry_date: dict[str, Optional[pd.Timestamp]] = {a: None for a in assets}

    for date, price_row in prices.iterrows():
        equity = cash + float((positions * price_row).sum())
        desired_positions = weights.loc[date] * equity / price_row

        for asset in assets:
            price = price_row[asset]
            current = positions[asset]
            desired = desired_positions[asset]

            if desired == current:
                continue

            cost = get_cost(asset)
            slip = get_slip(asset)

            # Handle position flip as close followed by open
            if current != 0 and desired != 0 and current * desired < 0:
                # Close existing
                trade_size = -current
                trade_price = price * (1 + slip if trade_size > 0 else 1 - slip)
                cash -= trade_size * trade_price
                cash -= cost

                pnl = (trade_price - entry_price[asset]) * current  # type: ignore[arg-type]
                trade_pnls.append(pnl)
                holding = (
                    (date - entry_date[asset]).days
                    if isinstance(date, pd.Timestamp)
                    else date - entry_date[asset]
                )  # type: ignore[arg-type]
                holding_periods.append(holding)
                trades.append(
                    {
                        "asset": asset,
                        "entry_date": entry_date[asset],
                        "exit_date": date,
                        "size": current,
                        "entry_price": entry_price[asset],
                        "exit_price": trade_price,
                        "pnl": pnl,
                        "holding_period": holding,
                    }
                )

                # Open new position
                trade_size = desired
                trade_price = price * (1 + slip if trade_size > 0 else 1 - slip)
                cash -= trade_size * trade_price
                cash -= cost

                entry_price[asset] = trade_price
                entry_date[asset] = date
                positions[asset] = desired
                continue

            # Simple change in position (including open or close)
            trade_size = desired - current
            trade_price = price * (1 + slip if trade_size > 0 else 1 - slip)
            cash -= trade_size * trade_price
            cash -= cost

            if current != 0:
                pnl = (trade_price - entry_price[asset]) * current  # type: ignore[arg-type]
                trade_pnls.append(pnl)
                holding = (
                    (date - entry_date[asset]).days
                    if isinstance(date, pd.Timestamp)
                    else date - entry_date[asset]
                )  # type: ignore[arg-type]
                holding_periods.append(holding)
                trades.append(
                    {
                        "asset": asset,
                        "entry_date": entry_date[asset],
                        "exit_date": date,
                        "size": current,
                        "entry_price": entry_price[asset],
                        "exit_price": trade_price,
                        "pnl": pnl,
                        "holding_period": holding,
                    }
                )

            if desired != 0:
                entry_price[asset] = trade_price
                entry_date[asset] = date
            else:
                entry_price[asset] = None
                entry_date[asset] = None

            positions[asset] = desired

        equity_curve.append({"date": date, "equity": cash + float((positions * price_row).sum())})

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
