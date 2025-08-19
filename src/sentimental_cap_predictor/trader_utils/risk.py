"""Risk management utilities.

This module currently provides an ``enforce_limits`` function which monitors
rolling profit and loss (PnL) and emits a trading state of ``"FLAT"`` when
predefined limits are breached.  Once a breach occurs the state remains
``"FLAT"`` for all subsequent observations.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable


def enforce_limits(
    pnl: Iterable[float],
    window: int,
    loss_limit: float,
    profit_limit: float | None = None,
) -> list[str]:
    """Monitor rolling PnL and emit trading state on limit breaches.

    Parameters
    ----------
    pnl:
        Iterable of profit and loss values over time.
    window:
        Number of recent observations to include in the rolling calculation.
    loss_limit:
        Maximum allowed cumulative loss over the rolling window. Must be
        positive.
    profit_limit:
        Optional profit target. If provided and the rolling PnL exceeds this
        value, the state transitions to ``"FLAT"``.  The default is ``None``
        which disables the profit check.

    Returns
    -------
    list[str]
        Sequence of trading states corresponding to each PnL observation.
        Values are either ``"LIVE"`` or ``"FLAT"``. Once ``"FLAT"`` it remains
        ``"FLAT"`` for all subsequent points.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if loss_limit <= 0:
        raise ValueError("loss_limit must be positive")
    if profit_limit is not None and profit_limit <= 0:
        raise ValueError("profit_limit must be positive when provided")

    recent = deque()
    rolling_total = 0.0
    flat = False
    states: list[str] = []

    for value in pnl:
        recent.append(value)
        rolling_total += value
        if len(recent) > window:
            rolling_total -= recent.popleft()

        if not flat:
            if rolling_total <= -loss_limit or (
                profit_limit is not None and rolling_total >= profit_limit
            ):
                flat = True

        states.append("FLAT" if flat else "LIVE")

    return states
