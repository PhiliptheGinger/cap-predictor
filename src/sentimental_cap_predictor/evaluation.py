from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Hard limits applied to strategy evaluation."""

    max_drawdown: float = 0.2  # 20%
    min_trades: int = 30


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute the annualised Sharpe ratio for a series of returns."""

    if returns.std(ddof=0) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0)


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown of an equity curve."""

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    metrics = {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "trade_count": len(trades),
    }
    return metrics


def objective(metrics: Dict[str, float]) -> float:
    """Simple objective function: risk-adjusted return via Sharpe ratio."""

    return metrics.get("sharpe_ratio", 0.0)


def passes_constraints(metrics: Dict[str, float], constraints: Constraints | None = None) -> bool:
    constraints = constraints or Constraints()
    if metrics.get("trade_count", 0) < constraints.min_trades:
        return False
    if abs(metrics.get("max_drawdown", 0)) > constraints.max_drawdown:
        return False
    return True
