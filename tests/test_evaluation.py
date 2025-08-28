import pandas as pd
import pytest

from sentimental_cap_predictor.evaluation import (
    Constraints,
    compute_metrics,
    objective,
    passes_constraints,
    rank_strategies,
)


def test_compute_metrics_includes_new_metrics():
    equity = pd.Series([1000, 1100, 1045, 1066], dtype=float)
    trades = pd.DataFrame({"size": [10, -5], "entry_price": [100, 105]})
    metrics = compute_metrics(equity, trades)
    assert "sortino_ratio" in metrics
    assert "volatility" in metrics
    assert "turnover" in metrics
    expected_turnover = (abs(10 * 100) + abs(-5 * 105)) / 1000
    assert metrics["turnover"] == expected_turnover


def test_objective_supports_expression():
    metrics = {"sharpe_ratio": 1.0, "max_drawdown": -0.1}
    value = objective(metrics, expression="sharpe_ratio + max_drawdown")
    assert value == pytest.approx(0.9)


def test_objective_rejects_malicious_expression():
    metrics = {"sharpe_ratio": 1.0}
    with pytest.raises(ValueError):
        objective(metrics, expression="__import__('os').system('echo hacked')")


def test_objective_supports_weights():
    metrics = {"sharpe_ratio": 1.0, "max_drawdown": -0.2}
    weights = {"sharpe_ratio": 1.0, "max_drawdown": 0.5}
    value = objective(metrics, weights=weights)
    assert value == pytest.approx(0.9)


def test_passes_constraints_configurable():
    metrics = {"trade_count": 40, "max_drawdown": -0.1, "volatility": 0.15}
    constraints = Constraints(max_drawdown=0.05)
    assert not passes_constraints(metrics, constraints)
    constraints = Constraints(max_drawdown=0.2, max_volatility=0.1)
    assert not passes_constraints(metrics, constraints)
    constraints = Constraints(max_drawdown=0.2, max_volatility=0.2)
    assert passes_constraints(metrics, constraints)


def test_rank_strategies_orders_by_objective():
    strategies = {
        "A": {"sharpe_ratio": 1.0, "max_drawdown": -0.1},
        "B": {"sharpe_ratio": 0.5, "max_drawdown": -0.05},
    }
    ranked = rank_strategies(strategies, expression="sharpe_ratio + max_drawdown")
    assert list(ranked["strategy"]) == ["A", "B"]


def test_rank_strategies_applies_constraints():
    strategies = {
        "good": {"sharpe_ratio": 1.0, "max_drawdown": -0.1, "trade_count": 20},
        "bad": {"sharpe_ratio": 2.0, "max_drawdown": -0.5, "trade_count": 5},
    }
    cons = Constraints(max_drawdown=0.2, min_trades=10)
    ranked = rank_strategies(strategies, constraints=cons)
    assert list(ranked["strategy"]) == ["good"]
