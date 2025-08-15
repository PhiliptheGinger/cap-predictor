import pandas as pd

from sentimental_cap_predictor.trader_utils.strategy_optimizer import (
    moving_average_crossover,
    random_search,
)


def test_moving_average_crossover_returns_float():
    prices = pd.Series([1, 2, 3, 4, 5], dtype=float)
    result = moving_average_crossover(prices, short_window=1, long_window=2)
    assert isinstance(result, float)


def test_random_search_returns_parameters():
    prices = pd.Series(range(1, 21), dtype=float)
    res = random_search(
        prices,
        iterations=10,
        short_range=(1, 3),
        long_range=(4, 6),
        seed=42,
    )
    assert res.short_window < res.long_window
    assert isinstance(res.total_return, float)
