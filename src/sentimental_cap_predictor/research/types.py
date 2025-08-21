"""Core type definitions used across research workflows.

This module centralises small data containers and protocols that are shared
between research experiments and backtesting utilities. These lightweight
structures provide a consistent interface for handling market and sentiment
inputs, passing configuration to backtests and capturing their results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from sentimental_cap_predictor import backtest_result
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


# Re-export BacktestResult from the core module so research utilities operate
# on the same unified dataclass used across the project.
BacktestResult = backtest_result.BacktestResult


class Strategy(Protocol):
    """Protocol that all trading strategies must implement."""

    def generate_signals(
        self, data: DataBundle, idea: Idea
    ) -> pd.Series | pd.DataFrame:
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
