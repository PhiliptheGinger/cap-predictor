from __future__ import annotations

"""Lightweight backtesting utilities for research workflows.

This module exposes a convenience function :func:`simple_backtester` that
creates a callable executing a basic daily backtest.  It operates on a
:class:`~sentimental_cap_predictor.research.types.DataBundle` and a
:class:`~sentimental_cap_predictor.research.idea_schema.Idea`, generating
trading signals via a provided :class:`~sentimental_cap_predictor.research.types.Strategy`.

The backtest assumes next-day ``open`` fills and models transaction costs via a
simple basis-point specification.  Common performance metrics are reported in a
:class:`~sentimental_cap_predictor.research.types.BacktestResult` dataclass.

``SmaSentStrategy`` is provided as a minimal reference implementation of the
``Strategy`` protocol combining price momentum with optional sentiment filters.
"""

from dataclasses import dataclass
from typing import Callable, List, Dict

import numpy as np
import pandas as pd

from .idea_schema import Idea
from .types import (
    BacktestContext,
    BacktestResult,
    DataBundle,
    Strategy,
    Trade,
)


@dataclass
class SmaSentStrategy:
    """Simple moving-average and sentiment crossover strategy.

    This reference strategy goes long when a short moving average is above a
    long moving average and optional sentiment scores are positive.
    The moving-average windows and sentiment column can be supplied via
    ``Idea.params`` using keys ``short_window``, ``long_window`` and
    ``sentiment_col`` respectively.
    """

    def generate_signals(self, data: DataBundle, idea: Idea) -> pd.Series:
        prices = data.prices.iloc[:, 0].astype(float)
        short_window = int(idea.params.get("short_window", 20))
        long_window = int(idea.params.get("long_window", 50))

        short_ma = prices.rolling(window=short_window, min_periods=1).mean()
        long_ma = prices.rolling(window=long_window, min_periods=1).mean()
        signals = (short_ma > long_ma).astype(float)

        if data.sentiment is not None and not data.sentiment.empty:
            sent_col = idea.params.get("sentiment_col")
            if sent_col and sent_col in data.sentiment:
                sentiment = data.sentiment[sent_col]
            else:
                sentiment = data.sentiment.iloc[:, 0]
            signals = signals.where(sentiment > 0, 0.0)

        return signals


def simple_backtester(strategy: Strategy) -> Callable[[DataBundle, Idea, BacktestContext], BacktestResult]:
    """Create a function that backtests ``strategy`` over provided data.

    The returned callable performs a daily backtest using ``DataBundle.prices``
    with trades executed at the next day's open.  Transaction costs are modelled
    as ``(fees_bps + slip_bps)/1e4`` per unit of turnover.
    """

    def _run(data: DataBundle, idea: Idea, ctx: BacktestContext | None = None) -> BacktestResult:
        ctx = ctx or BacktestContext()

        prices = data.prices.copy()
        open_col = next((c for c in prices.columns if c.lower().startswith("open")), None)
        opens = prices[open_col] if open_col else prices.iloc[:, 0]
        opens = opens.astype(float)

        signals = strategy.generate_signals(data, idea).reindex(opens.index).fillna(0.0).astype(float)
        positions = signals.shift(1).fillna(0.0)
        returns = opens.pct_change().shift(-1).fillna(0.0)

        trades = positions.diff().abs().fillna(0.0)
        cost_per_trade = (ctx.fees_bps + ctx.slip_bps) / 1e4
        costs = trades * cost_per_trade
        strategy_returns = positions * returns - costs
        equity_curve = (1.0 + strategy_returns).cumprod()

        trade_list: List[Trade] = []
        symbol = data.meta.get("ticker", "")
        pos_diff = positions.diff().fillna(positions)
        for ts, change in pos_diff[pos_diff != 0].items():
            side = "buy" if change > 0 else "sell"
            fees = float(abs(change) * cost_per_trade)
            note = str(ts)
            trade_list.append(
                Trade(
                    symbol=symbol,
                    side=side,
                    qty=float(change),
                    price=float(opens.loc[ts]),
                    fees=fees,
                    note=note,
                )
            )

        n_days = len(opens)
        cagr = float(equity_curve.iloc[-1] ** (252 / n_days) - 1) if n_days > 0 else 0.0
        sharpe = float(np.sqrt(252) * strategy_returns.mean() / strategy_returns.std(ddof=0)) if strategy_returns.std(ddof=0) > 0 else np.nan
        cumulative_max = equity_curve.cummax()
        drawdown = equity_curve / cumulative_max - 1.0
        maxdd = float(drawdown.min())
        turnover = float(trades.sum())

        metrics: Dict[str, float] = {
            "CAGR": cagr,
            "Sharpe": sharpe,
            "MaxDD": maxdd,
            "Turnover": turnover,
        }

        return BacktestResult(
            idea_name=idea.name,
            equity_curve=equity_curve,
            trades=trade_list,
            positions=positions,
            metrics=metrics,
            artifacts={},
        )

    return _run
