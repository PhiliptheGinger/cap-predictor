import pandas as pd
import pytest

from sentimental_cap_predictor.research.engine import simple_backtester
from sentimental_cap_predictor.research.types import BacktestContext, DataBundle
from sentimental_cap_predictor.research.idea_schema import Idea


class AlwaysLong:
    def generate_signals(self, data, idea):
        return pd.Series(1.0, index=data.prices.index)


def test_constant_weight_pnl_and_cost_impact():
    index = pd.date_range('2020-01-01', periods=3, freq='D')
    prices = pd.DataFrame({'open': [100, 100, 100], 'close': [101, 101, 101]}, index=index)
    data = DataBundle(prices=prices)
    idea = Idea(name='long')
    ctx = BacktestContext(fees_bps=1, slip_bps=2)

    backtester = simple_backtester(AlwaysLong())
    result = backtester(data, idea, ctx)

    cost = (ctx.fees_bps + ctx.slip_bps) / 1e4
    expected_equity = (1.0) * (1 + 0) * (1 + 0.01 - cost) * (1 + 0.01)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity)
    assert len(result.trades) == 1
    assert result.trades[0].fees == pytest.approx(cost)

