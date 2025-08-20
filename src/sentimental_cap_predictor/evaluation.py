from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Hard limits applied to strategy evaluation.

    Parameters can be set to ``None`` to disable a particular constraint.
    """

    max_drawdown: float | None = 0.2  # 20%
    min_trades: int | None = 30
    max_volatility: float | None = None
    max_turnover: float | None = None


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute the annualised Sharpe ratio for a series of returns."""

    if returns.std(ddof=0) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute the annualised Sortino ratio for a series of returns."""

    downside = returns[returns < 0]
    if downside.std(ddof=0) == 0:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=0)


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Return the annualised volatility of ``returns``."""

    if returns.empty:
        return 0.0
    return returns.std(ddof=0) * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown of an equity curve."""

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    turnover = 0.0
    if not trades.empty:
        turnover = (trades["size"].abs() * trades["entry_price"].abs()).sum() / float(
            equity.iloc[0]
        )
    metrics = {
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "volatility": annualised_volatility(returns),
        "turnover": turnover,
        "max_drawdown": max_drawdown(equity),
        "trade_count": len(trades),
    }
    return metrics


def objective(
    metrics: Dict[str, float],
    *,
    expression: str | None = None,
    weights: Dict[str, float] | None = None,
) -> float:
    """Evaluate a composite objective based on ``metrics``.

    Parameters
    ----------
    metrics:
        Dictionary of metric values.
    expression:
        Optional Python expression combining metric names, e.g. ``"sharpe_ratio + 0.5*max_drawdown"``.
        ``max_drawdown`` is typically negative, so adding it penalises the objective.
    weights:
        Alternatively, provide a mapping of metric weights.  The objective becomes
        ``sum(weight * metric)``.

    Returns
    -------
    float
        The evaluated objective value.
    """

    if expression:
        local = {k: float(v) for k, v in metrics.items()}
        try:
            return float(eval(expression, {"__builtins__": {}}, local))
        except Exception:
            return float("nan")
    if weights:
        return float(sum(metrics.get(k, 0.0) * w for k, w in weights.items()))
    return metrics.get("sharpe_ratio", 0.0)


def passes_constraints(metrics: Dict[str, float], constraints: Constraints | None = None) -> bool:
    constraints = constraints or Constraints()
    if constraints.min_trades is not None and metrics.get("trade_count", 0) < constraints.min_trades:
        return False
    if (
        constraints.max_drawdown is not None
        and abs(metrics.get("max_drawdown", 0)) > constraints.max_drawdown
    ):
        return False
    if (
        constraints.max_volatility is not None
        and metrics.get("volatility", 0.0) > constraints.max_volatility
    ):
        return False
    if (
        constraints.max_turnover is not None
        and metrics.get("turnover", 0.0) > constraints.max_turnover
    ):
        return False
    return True


def rank_strategies(
    strategies: Dict[str, Dict[str, float]] | Iterable[tuple[str, Dict[str, float]]],
    *,
    expression: str | None = None,
    weights: Dict[str, float] | None = None,
    constraints: Constraints | None = None,
) -> pd.DataFrame:
    """Rank ``strategies`` according to a composite objective.

    Parameters
    ----------
    strategies:
        Mapping or iterable of (name, metrics) pairs.
    expression, weights:
        Passed to :func:`objective`.
    constraints:
        Optional :class:`Constraints` limiting admissible strategies.

    Returns
    -------
    DataFrame
        DataFrame sorted by objective value in descending order.  Contains the
        objective and all metric values for each strategy that passes the
        constraints.
    """

    if isinstance(strategies, dict):
        items = strategies.items()
    else:
        items = strategies
    rows = []
    for name, mets in items:
        if constraints and not passes_constraints(mets, constraints):
            continue
        obj = objective(mets, expression=expression, weights=weights)
        row = {"strategy": name, "objective": obj}
        row.update(mets)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("objective", ascending=False).reset_index(drop=True)
