import numpy as np
import pandas as pd
import pytest

from sentimental_cap_predictor.research import objectives
from sentimental_cap_predictor.research.types import BacktestResult


def _make_result(equity: pd.Series) -> BacktestResult:
    return BacktestResult(
        trades=pd.DataFrame(),
        equity_curve=equity,
        metrics={},
        parameters={},
        trade_pnls=pd.Series(dtype=float),
        holding_periods=pd.Series(dtype=float),
    )


def test_sharpe_and_sortino_use_returns():
    equity = pd.Series([1.0, 1.02, 1.0098, 1.040094, 1.019292])
    result = _make_result(equity)
    returns = result.return_series()

    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    downside = returns[returns < 0].std(ddof=0)
    expected_sortino = np.sqrt(252) * returns.mean() / downside

    assert objectives.sharpe(result) == pytest.approx(expected_sharpe)
    assert objectives.sortino(result) == pytest.approx(expected_sortino)


def test_calmar_uses_equity_curve():
    equity = pd.Series([1.0, 1.02, 1.0098, 1.040094, 1.019292])
    result = _make_result(equity)

    n = len(equity)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / n) - 1
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    maxdd = abs(drawdown.min())
    expected_calmar = cagr / maxdd

    assert objectives.calmar(result) == pytest.approx(expected_calmar)
