from __future__ import annotations

from itertools import product
import random
from typing import Callable, Iterable, Any, Tuple

from .search_space import SearchSpace, ParamSpec


def _grid_values(spec: ParamSpec) -> list[Any]:
    """Return the grid values for a parameter specification.

    Numeric parameters are expanded to a three point grid (low, mid, high)
    while categorical parameters simply use their specified choices.
    """

    if spec.kind in {"int", "float"}:
        if spec.bounds is None:
            raise ValueError("Numeric parameters require bounds")
        low, high = spec.bounds
        mid = (low + high) / 2
        values = [low, mid, high]
        if spec.kind == "int":
            values = [int(round(v)) for v in values]
        return values
    if spec.kind == "choice":
        if spec.choices is None:
            raise ValueError("Choice parameters require a choices list")
        return list(spec.choices)
    raise ValueError(f"Unknown parameter kind: {spec.kind}")


def grid_optimize(
    base_idea: dict,
    backtest: Callable[[dict, Any, Any], Any],
    data: Any,
    ctx: Any,
    space: SearchSpace,
    objective: Callable[[Any], float],
    constraints: Iterable[Callable[[Any], bool]] = [],
    max_points: int = 81,
) -> Tuple[dict, Any]:
    """Perform a simple grid search over the provided parameter space.

    Parameters
    ----------
    base_idea:
        Base configuration to update for each point in the search space.
    backtest:
        Callable that evaluates an idea given ``data`` and ``ctx``.
    data, ctx:
        Additional arguments forwarded to ``backtest``.
    space:
        Mapping of parameter names to their :class:`ParamSpec`.
    objective:
        Callable returning a numeric score from the backtest result. Higher is
        better.
    constraints:
        Iterable of callables that take the backtest result and return ``True``
        if the configuration is valid.
    max_points:
        Maximum number of grid points to evaluate. If the full Cartesian grid
        exceeds this number, a random subset is evaluated.

    Returns
    -------
    tuple
        ``(best_idea, best_result)`` where ``best_idea`` is the configuration
        yielding the highest objective value.
    """

    # Build the grid for each parameter
    names = list(space.keys())
    grids = [_grid_values(space[name]) for name in names]

    # Cartesian product of all parameter grids
    candidates = list(product(*grids)) if grids else [tuple()]

    # Randomly subsample if too many points
    if len(candidates) > max_points:
        rng = random.Random(0)
        candidates = rng.sample(candidates, max_points)

    best_idea = dict(base_idea)
    best_value = float("-inf")
    best_result = None

    for combo in candidates:
        idea = dict(base_idea)
        idea.update({name: value for name, value in zip(names, combo)})

        result = backtest(idea, data, ctx)
        if constraints and not all(c(result) for c in constraints):
            continue

        value = objective(result)
        if value > best_value:
            best_value = value
            best_idea = idea
            best_result = result

    return best_idea, best_result
