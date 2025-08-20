"""Core type definitions used across research workflows.

This module centralises small data containers and protocols that are shared
between research experiments and backtesting utilities. These lightweight
structures provide a consistent interface for handling market and sentiment
inputs, passing configuration to backtests and capturing their results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import pandas as pd

from sentimental_cap_predictor.data_bundle import DataBundle
from .idea_schema import Idea


@dataclass
class BacktestContext:
    """Runtime configuration for a backtest session.

    Attributes
    ----------
    fees_bps:
        Commission fees expressed in basis points.
    slip_bps:
        Slippage assumption expressed in basis points.
    seed:
        Random seed used for any stochastic components.
    calendar:
        Trading calendar name defining market sessions.
    """

    fees_bps: float = 1.0
    slip_bps: float = 2.0
    seed: int = 42
    calendar: str = "NYSE"


@dataclass
class Trade:
    """Representation of a single executed trade."""

    symbol: str
    side: str
    qty: float
    price: float
    fees: float = 0.0
    note: str = ""


@dataclass
class BacktestResult:
    """Outcome of a backtesting run.

    Attributes
    ----------
    idea_name:
        Name of the idea or strategy being evaluated.
    equity_curve:
        ``pandas.Series`` representing cumulative equity over time.
    trades:
        Executed trades captured during the backtest.
    positions:
        Position sizes through time as a ``Series`` or ``DataFrame`` when
        multiple instruments are involved.
    metrics:
        Dictionary of performance metrics such as CAGR or Sharpe ratio.
    artifacts:
        Additional objects produced during the run (plots, tables, etc.).
    """

    idea_name: str
    equity_curve: pd.Series
    trades: List[Trade]
    positions: pd.Series | pd.DataFrame
    metrics: Dict[str, Any]
    artifacts: Dict[str, float | Any]


class Strategy(Protocol):
    """Protocol that all trading strategies must implement."""

    def generate_signals(self, data: DataBundle, idea: Idea) -> pd.Series | pd.DataFrame:
        """Return trading signals for ``idea`` based on ``data``.

        Parameters
        ----------
        data:
            Bundle of market data and optional sentiment features.
        idea:
            Research idea describing the strategy configuration.

        Returns
        -------
        pandas.Series | pandas.DataFrame
            Generated trading signals aligned with the ``prices`` index.
        """
        ...
