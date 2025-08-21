"""Lightweight backtesting utilities for research workflows.

This module exposes a convenience function :func:`simple_backtester` that
creates a callable executing a basic daily backtest. It operates on a
:class:`~sentimental_cap_predictor.research.types.DataBundle` and a
:class:`~sentimental_cap_predictor.research.idea_schema.Idea`, generating
trading signals via a provided
:class:`~sentimental_cap_predictor.research.types.Strategy`.

The backtest assumes next-day ``open`` fills and models transaction costs via a
simple basis-point specification. Common performance metrics are reported in a
:class:`~sentimental_cap_predictor.research.types.BacktestResult` dataclass.

``SmaSentStrategy`` is provided as a minimal reference implementation of the
``Strategy`` protocol combining price momentum with optional sentiment filters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from sentimental_cap_predictor.data_bundle import DataBundle

from .idea_schema import Idea
from .types import BacktestContext, BacktestResult, Strategy, Trade


@dataclass
class SmaSentStrategy:
    """Price above moving-average with optional sentiment filter.

    The strategy goes long when the ``close`` price is above a simple moving
    average of length ``ma_window``.  If sentiment data are supplied the signal
    is only active when the sentiment value exceeds ``sent_thr``.  Both
    parameters are read from ``Idea.params``.
    """

    def generate_signals(self, data: DataBundle, idea: Idea) -> pd.Series:
        prices = data.prices.iloc[:, 0].astype(float)
        ma_window = int(idea.params.get("ma_window", 20))
        sent_thr = float(idea.params.get("sent_thr", 0.0))

        ma = prices.rolling(window=ma_window, min_periods=1).mean()
        signals = (prices > ma).astype(float)

        if data.sentiment is not None and not data.sentiment.empty:
            sentiment = data.sentiment.iloc[:, 0].astype(float)
            signals = signals.where(sentiment > sent_thr, 0.0)

        return signals


def simple_backtester(
    strategy: Strategy,
) -> Callable[[DataBundle, Idea, BacktestContext], BacktestResult]:
    """Create a function that backtests ``strategy`` over provided data.

    The returned callable performs a daily backtest using ``DataBundle.prices``
    with trades executed at the next day's open. Transaction costs are modelled
    as ``(fees_bps + slip_bps)/1e4`` per unit of turnover.
    """

    def _run(
        data: DataBundle, idea: Idea, ctx: BacktestContext | None = None
    ) -> BacktestResult:
        ctx = ctx or BacktestContext()

        prices = data.prices.copy()
        open_col = next(
            (c for c in prices.columns if c.lower().startswith("open")), None
        )
        close_col = next(
            (c for c in prices.columns if c.lower().startswith("close")), None
        )
        opens = prices[open_col] if open_col else prices.iloc[:, 0]
        closes = prices[close_col] if close_col else prices.iloc[:, -1]
        opens = opens.astype(float)
        closes = closes.astype(float)

        signals = (
            strategy.generate_signals(data, idea)
            .reindex(opens.index)
            .fillna(0.0)
            .astype(float)
        )
        positions = signals.shift(1).fillna(0.0)
        returns = (closes / opens - 1.0).fillna(0.0)

        pos_diff = positions.diff().fillna(positions)
        trades = pos_diff.abs()
        cost_per_trade = (ctx.fees_bps + ctx.slip_bps) / 1e4
        costs = trades * cost_per_trade
        strategy_returns = positions * returns - costs
        equity_curve = (1.0 + strategy_returns).cumprod()

        trade_list: List[Trade] = []
        trade_pnls: List[float] = []
        holding_periods: List[float] = []
        symbol = (data.metadata or {}).get("ticker", "")
        position = 0.0
        entry_price: float | None = None
        entry_ts: pd.Timestamp | int | None = None

        for ts, change in pos_diff[pos_diff != 0].items():
            price = float(opens.loc[ts])
            side = "buy" if change > 0 else "sell"
            fees = float(abs(change) * cost_per_trade)
            note = str(ts)

            pnl = 0.0
            holding = 0.0
            closing = position != 0 and (position + change) * position <= 0
            if closing and entry_price is not None and entry_ts is not None:
                pnl = (price - entry_price) * position
                delta = ts - entry_ts
                holding = (
                    delta.days if isinstance(delta, pd.Timedelta) else float(delta)
                )

            trade_list.append(
                Trade(
                    symbol=symbol,
                    side=side,
                    qty=float(change),
                    price=price,
                    fees=fees,
                    note=note,
                )
            )
            trade_pnls.append(float(pnl))
            holding_periods.append(float(holding))

            position += change
            if position != 0 and closing:
                # flipped position: treat current price as new entry
                entry_price = price
                entry_ts = ts
            elif position != 0 and entry_price is None:
                entry_price = price
                entry_ts = ts
            elif position == 0:
                entry_price = None
                entry_ts = None

        n_days = len(opens)
        if n_days > 0:
            cagr = float(equity_curve.iloc[-1] ** (252 / n_days) - 1)
        else:
            cagr = 0.0
        ret_std = strategy_returns.std(ddof=0)
        sharpe = (
            float(np.sqrt(252) * strategy_returns.mean() / ret_std)
            if ret_std > 0
            else np.nan
        )
        vol = float(ret_std * np.sqrt(252))
        cumulative_max = equity_curve.cummax()
        drawdown = equity_curve / cumulative_max - 1.0
        maxdd = float(drawdown.min())
        mean_turnover = float(trades.mean())
        trade_count = float((trades > 0).sum())

        metrics: Dict[str, float] = {
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Vol": vol,
            "MaxDD": maxdd,
            "Turnover": mean_turnover,
            "TradeCount": trade_count,
        }

        trades_df = (
            pd.DataFrame([asdict(t) for t in trade_list])
            if trade_list
            else pd.DataFrame(
                columns=["symbol", "side", "qty", "price", "fees", "note"]
            )
        )
        trade_pnls_series = pd.Series(trade_pnls, dtype=float)
        holding_series = pd.Series(holding_periods, dtype=float)
        params = {"idea_name": idea.name, "positions": positions}
        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_curve,
            metrics=metrics,
            parameters=params,
            trade_pnls=trade_pnls_series,
            holding_periods=holding_series,
        )

    return _run
