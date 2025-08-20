from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class BacktestResult:
    """Holds the outputs of a backtest run."""

    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]
