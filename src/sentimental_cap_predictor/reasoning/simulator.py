from __future__ import annotations

"""Very small deterministic simulator for trend and force states."""

from typing import Dict, Tuple

State = Dict[str, str]


def step(state: State) -> Tuple[State, str]:
    """Advance one step in the simulation.

    Parameters
    ----------
    state:
        Mapping with keys ``"trend"`` and ``"force"``. ``"trend`` may be
        ``"up"`` or ``"down"``; ``"force"`` may be ``"weak"`` or ``"strong"``.

    Returns
    -------
    tuple[State, str]
        ``(next_state, narration)`` describing the updated state and a short
        explanation of what occurred.

    Raises
    ------
    KeyError
        If required keys are missing.
    ValueError
        If values are outside the allowed sets.
    """

    trend = state["trend"]
    force = state["force"]

    if trend not in {"up", "down"}:
        raise ValueError("trend must be 'up' or 'down'")
    if force not in {"weak", "strong"}:
        raise ValueError("force must be 'weak' or 'strong'")

    if force == "strong":
        next_trend = trend
        next_force = "weak"
        narration = f"Strong force keeps the trend {next_trend}."
    else:
        next_trend = "down" if trend == "up" else "up"
        next_force = "strong"
        narration = f"Weak force allows the trend to reverse to {next_trend}."

    next_state: State = {"trend": next_trend, "force": next_force}
    return next_state, narration
