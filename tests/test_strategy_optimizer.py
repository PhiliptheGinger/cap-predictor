import pandas as pd
import pytest

from sentimental_cap_predictor.trader_utils.strategy_optimizer import (
    moving_average_crossover,
    random_search,
    walk_forward_eval,
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
        lambda_drawdown=0.5,
    )
    assert res.short_window < res.long_window
    assert isinstance(res.score, float)
    assert isinstance(res.mean_return, float)
    assert isinstance(res.mean_drawdown, float)


def test_random_search_invalid_iterations():
    prices = pd.Series(range(10), dtype=float)
    with pytest.raises(ValueError):
        random_search(prices, iterations=0)


def test_random_search_invalid_short_range():
    prices = pd.Series(range(10), dtype=float)
    with pytest.raises(ValueError):
        random_search(
            prices,
            iterations=1,
            short_range=(5, 2),
            long_range=(3, 10),
        )


def test_random_search_invalid_long_range():
    prices = pd.Series(range(10), dtype=float)
    with pytest.raises(ValueError):
        random_search(
            prices,
            iterations=1,
            short_range=(1, 2),
            long_range=(5, 4),
        )


def test_walk_forward_eval_outputs_floats():
    prices = pd.Series(range(1, 21), dtype=float)
    mean_ret, mean_dd = walk_forward_eval(prices, 1, 2, n_splits=3)
    assert isinstance(mean_ret, float)
    assert isinstance(mean_dd, float)
