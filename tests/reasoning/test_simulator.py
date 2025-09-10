import pytest

from sentimental_cap_predictor.reasoning.simulator import step


def test_step_strong_force_keeps_trend() -> None:
    state = {"trend": "up", "force": "strong"}
    next_state, narration = step(state)
    assert next_state == {"trend": "up", "force": "weak"}
    assert "Strong force keeps the trend up" in narration


def test_step_weak_force_reverses_trend() -> None:
    state = {"trend": "down", "force": "weak"}
    next_state, narration = step(state)
    assert next_state == {"trend": "up", "force": "strong"}
    assert "reverse" in narration and "up" in narration


def test_step_invalid_values() -> None:
    with pytest.raises(ValueError):
        step({"trend": "sideways", "force": "strong"})
    with pytest.raises(ValueError):
        step({"trend": "up", "force": "medium"})
