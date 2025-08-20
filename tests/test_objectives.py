import numpy as np
import pandas as pd
import pytest

from sentimental_cap_predictor.research import objectives
from sentimental_cap_predictor.research.types import BacktestResult


def _make_result(returns: pd.Series, equity: pd.Series) -> BacktestResult:
    return BacktestResult(
        idea_name="test",
        equity_curve=equity,
        trades=[],
        positions=pd.Series(dtype=float),
        metrics={},
        artifacts={"strat_ret": returns},
    )


def test_sharpe_and_sortino_use_artifact_returns():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    equity = pd.Series(1.0, index=returns.index)  # flat equity
    result = _make_result(returns, equity)

    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    downside = returns[returns < 0].std(ddof=0)
    expected_sortino = np.sqrt(252) * returns.mean() / downside

    assert objectives.sharpe(result) == pytest.approx(expected_sharpe)
    assert objectives.sortino(result) == pytest.approx(expected_sortino)


def test_calmar_uses_equity_curve():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    equity = (1 + returns).cumprod()
    result = _make_result(returns, equity)

    n = len(equity)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / n) - 1
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    maxdd = abs(drawdown.min())
    expected_calmar = cagr / maxdd

    assert objectives.calmar(result) == pytest.approx(expected_calmar)
