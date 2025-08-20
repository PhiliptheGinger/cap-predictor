"""Constraint helper functions used to filter backtest results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .types import BacktestResult


def max_drawdown(
    result: "BacktestResult",
    limit: float = 0.25,
) -> tuple[bool, str]:
    """Check that maximum drawdown does not exceed ``limit``."""

    equity = result.equity_curve.astype(float)
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    maxdd = float(drawdown.min())
    ok = maxdd >= -limit
    status = "within" if ok else "exceeds"
    msg = f"Max drawdown {maxdd:.2%} {status} limit {limit:.2%}"
    return ok, msg


def min_trades(result: "BacktestResult", n: int = 30) -> tuple[bool, str]:
    """Ensure at least ``n`` trades were executed."""

    count = len(result.trades)
    ok = count >= n
    msg = f"{count} trades {'meets' if ok else 'below'} minimum {n}"
    return ok, msg


def max_turnover(
    result: "BacktestResult",
    limit: float = 5.0,
) -> tuple[bool, str]:
    """Check that portfolio turnover is within ``limit``."""

    positions = result.parameters.get("positions")
    if positions is None:
        return True, "No positions data"

    if isinstance(positions, pd.DataFrame):
        turnover = float(positions.diff().abs().sum().sum())
    else:
        turnover = float(positions.diff().abs().sum())

    ok = turnover <= limit
    status = "within" if ok else "exceeds"
    msg = f"Turnover {turnover:.2f} {status} limit {limit:.2f}"
    return ok, msg
