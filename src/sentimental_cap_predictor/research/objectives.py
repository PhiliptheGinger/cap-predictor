"""Objective functions evaluating backtest performance.

These helpers operate on
:class:`~sentimental_cap_predictor.research.types.BacktestResult` instances
and compute common risk-adjusted metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .types import BacktestResult


def _returns(result: "BacktestResult") -> pd.Series:
    """Return daily strategy returns.

    Returns are derived from the equity curve of ``result``.
    """

    rets = result.return_series().astype(float)
    return rets.dropna()


def sharpe(result: "BacktestResult") -> float:
    """Annualised Sharpe ratio of ``result``."""

    rets = _returns(result)
    vol = rets.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float(np.sqrt(252) * rets.mean() / vol)


def sortino(result: "BacktestResult") -> float:
    """Annualised Sortino ratio using downside deviation."""

    rets = _returns(result)
    downside = rets[rets < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return float("nan")
    return float(np.sqrt(252) * rets.mean() / dd)


def calmar(result: "BacktestResult") -> float:
    """Calmar ratio defined as CAGR divided by max drawdown."""

    equity = result.equity_curve.astype(float)
    n = len(equity)
    if n == 0:
        return float("nan")
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (252 / n) - 1)
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    maxdd = float(drawdown.min())
    if maxdd == 0:
        return float("nan")
    return cagr / abs(maxdd)
