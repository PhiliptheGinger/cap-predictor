from types import SimpleNamespace

from research.optimize import grid_optimize
from research.search_space import ParamSpec


def test_optimizer_selects_best_sharpe():
    space = {'p': ParamSpec(kind='choice', choices=[1, 2, 3])}

    def backtest(idea, data, ctx):
        return SimpleNamespace(metrics={'Sharpe': idea['p']})

    def objective(result):
        return result.metrics['Sharpe']

    best_idea, _ = grid_optimize({}, backtest, None, None, space, objective)
    assert best_idea['p'] == 3

