import pandas as pd

from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.strategy import BuyAndHoldStrategy
from sentimental_cap_predictor.backtester import backtest


def test_buy_and_hold_generates_weight_dataframe():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [1, 2, 3]}, index=index)
    bundle = DataBundle(prices=prices).validate()

    strategy = BuyAndHoldStrategy()
    weights = strategy.generate_signals(bundle)

    assert list(weights.columns) == ["AAPL"]
    pd.testing.assert_frame_equal(weights, pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=index))


def test_backtester_interprets_weights():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"AAPL": [1, 2, 3]}, index=index)
    bundle = DataBundle(prices=prices).validate()

    strategy = BuyAndHoldStrategy()
    result = backtest(bundle, strategy, initial_capital=1_000.0)

    assert result.equity_curve.iloc[-1] == 3_000.0
