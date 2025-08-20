import pandas as pd
import pytest

from sentimental_cap_predictor.backtester import backtest
from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.strategy import StrategyIdea


class DummyStrategy(StrategyIdea):
    """Strategy returning pre-defined weights used for testing."""

    def __init__(self, weight_df: pd.DataFrame):
        super().__init__(None)
        self._weights = weight_df

    def generate_signals(self, data: DataBundle) -> pd.DataFrame:
        return self._weights


def test_multi_asset_next_bar_and_costs():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            ("AAPL", "close"): [10, 11, 12],
            ("MSFT", "close"): [20, 19, 18],
        },
        index=index,
    )
    bundle = DataBundle(prices=prices)

    weights = pd.DataFrame(
        [[1, -1], [0, 0], [0, 0]], index=index, columns=["AAPL", "MSFT"]
    )
    strat = DummyStrategy(weights)

    result = backtest(
        bundle,
        strat,
        initial_capital=1_000.0,
        cost_per_trade={"AAPL": 1.0, "MSFT": 2.0},
        slippage={"AAPL": 0.01, "MSFT": 0.02},
    )

    trades = result.trades
    assert set(trades["asset"]) == {"AAPL", "MSFT"}
    # Signals executed on next bar
    assert (trades["entry_date"] == index[1]).all()
    # Short positions allowed
    assert trades[trades["asset"] == "MSFT"]["size"].iloc[0] < 0
    # Fill prices include slippage
    aapl_trade = trades[trades["asset"] == "AAPL"].iloc[0]
    assert aapl_trade["entry_price"] == pytest.approx(11 * 1.01)
    assert aapl_trade["exit_price"] == pytest.approx(12 * 0.99)
    # Portfolio equity after closing positions
    assert result.equity_curve.iloc[-1] == pytest.approx(1077.6842105263)


def test_backtest_generates_metrics_and_trades():
    index = pd.date_range('2024-01-01', periods=3, freq='D')
    prices = pd.DataFrame({('AAPL', 'close'): [10, 11, 12]}, index=index)
    bundle = DataBundle(prices=prices)

    weights = pd.DataFrame([1.0, 0.0, 0.0], index=index, columns=['AAPL'])
    strat = DummyStrategy(weights)

    result = backtest(bundle, strat, initial_capital=1_000.0)

    # One completed trade with expected dates and pnl
    trades = result.trades
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade['entry_date'] == index[1]
    assert trade['exit_date'] == index[2]
    assert trade['pnl'] == pytest.approx(90.9091, rel=1e-4)

    # Metrics reflect equity growth and trade count
    assert result.metrics['total_return'] == pytest.approx(0.090909, rel=1e-4)
    assert result.metrics['trade_count'] == 1

