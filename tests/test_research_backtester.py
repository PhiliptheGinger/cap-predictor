import pandas as pd

from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.research.engine import simple_backtester
from sentimental_cap_predictor.research.idea_schema import Idea


class DummyStrategy:
    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, data: DataBundle, idea: Idea) -> pd.Series:
        return self._signals


def test_simple_backtester_trades_and_equity():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {"open": [10, 11, 12, 11], "close": [11, 12, 11, 12]}, index=dates
    )
    bundle = DataBundle(prices=prices, metadata={"ticker": "XYZ"})

    signals = pd.Series([1, 0, 0, 0], index=dates, dtype=float)
    strategy = DummyStrategy(signals)

    backtester = simple_backtester(strategy)
    result = backtester(bundle, Idea(name="dummy"))

    cost = (1.0 + 2.0) / 1e4
    opens = prices["open"].astype(float)
    closes = prices["close"].astype(float)
    positions = signals.shift(1).fillna(0.0)
    returns = (closes / opens - 1.0).fillna(0.0)
    trades = positions.diff().fillna(positions).abs()
    strategy_returns = positions * returns - trades * cost
    expected_equity = (1.0 + strategy_returns).cumprod()

    expected_trades = pd.DataFrame(
        {
            "symbol": ["XYZ", "XYZ"],
            "side": ["buy", "sell"],
            "qty": [1.0, -1.0],
            "price": [11.0, 12.0],
            "fees": [cost, cost],
            "note": [str(dates[1]), str(dates[2])],
        }
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    pd.testing.assert_frame_equal(
        result.trades.reset_index(drop=True),
        expected_trades,
    )
    expected_pnls = pd.Series([0.0, 1.0])
    expected_holding = pd.Series([0.0, 1.0])
    pd.testing.assert_series_equal(
        result.trade_pnls.reset_index(drop=True), expected_pnls
    )
    pd.testing.assert_series_equal(
        result.holding_periods.reset_index(drop=True), expected_holding
    )
