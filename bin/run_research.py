#!/usr/bin/env python
"""CLI utility for running research backtests and grid searches.

This script loads market data, evaluates a trading strategy either via a
straightforward backtest or an exhaustive grid search over a parameter
space and logs resulting metrics using the project tracker.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, Iterable

from sentimental_cap_predictor.data.loader import make_bundle
from sentimental_cap_predictor.research.engine import simple_backtester
from sentimental_cap_predictor.research.idea_schema import Idea
from sentimental_cap_predictor.research.optimize import grid_optimize
from sentimental_cap_predictor.research.sandbox import run_strategy_source
from sentimental_cap_predictor.research.search_space import ParamSpec, SearchSpace
from sentimental_cap_predictor.research.tracking import get_tracker
from sentimental_cap_predictor.research.types import BacktestContext, DataBundle


DEFAULT_OBJECTIVE = (
    "sentimental_cap_predictor.research.objectives.sharpe"
)


def _import_from(path: str) -> Callable:
    """Import and return the object at ``path``."""

    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _load_search_space(path: str) -> SearchSpace:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    space: SearchSpace = {}
    for name, cfg in raw.items():
        space[name] = ParamSpec(
            kind=cfg["kind"],
            bounds=tuple(cfg["bounds"]) if "bounds" in cfg else None,
            choices=cfg.get("choices"),
        )
    return space


def _split_bundle(bundle: DataBundle, split: float) -> tuple[DataBundle, DataBundle]:
    n = len(bundle.prices)
    cut = int(n * split)

    def _slice(df):
        if df is None or getattr(df, "empty", False):
            return df, df
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    prices_tr, prices_te = _slice(bundle.prices)
    sent_tr, sent_te = _slice(bundle.sentiment)
    fund_tr, fund_te = _slice(bundle.fundamentals)

    meta_tr = dict(bundle.meta)
    meta_te = dict(bundle.meta)
    return (
        DataBundle(prices_tr, sent_tr, fund_tr, meta_tr),
        DataBundle(prices_te, sent_te, fund_te, meta_te),
    )


def _build_backtest_fn(
    strategy_class: str | None,
    source: str | None,
) -> Callable[[dict, DataBundle, BacktestContext], object]:
    if source is not None:
        code = Path(source).read_text(encoding="utf-8")

        def _backtest(params: dict, data: DataBundle, ctx: BacktestContext):
            idea = Idea(name="idea", params=params)
            return run_strategy_source(code, data, idea, ctx)

        return _backtest

    if strategy_class is None:
        raise ValueError("Either strategy_class or source must be provided")
    cls = _import_from(strategy_class)

    def _backtest(params: dict, data: DataBundle, ctx: BacktestContext):
        strategy = cls()
        backtester = simple_backtester(strategy)
        idea = Idea(name="idea", params=params)
        return backtester(data, idea, ctx)

    return _backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research experiments")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--start", required=True, help="Start date")
    parser.add_argument("--end", required=True, help="End date")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strategy-class", help="Import path to Strategy class")
    group.add_argument("--source", help="Path to Python file defining `strategy`")
    parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        help="Import path to objective function",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Import path to constraint function (may repeat)",
    )
    parser.add_argument("--search-space", help="JSON file describing parameter search space")
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/test split fraction",
    )
    args = parser.parse_args()

    bundle = make_bundle(args.ticker, args.start, args.end)
    train_bundle, test_bundle = _split_bundle(bundle, args.split)

    objective = _import_from(args.objective)
    constraints: Iterable[Callable] = [
        _import_from(path) for path in args.constraint
    ]

    backtest = _build_backtest_fn(args.strategy_class, args.source)
    ctx = BacktestContext()

    tracker = get_tracker()
    tracker.start_run()
    try:
        if args.search_space:
            space = _load_search_space(args.search_space)
            base_idea: dict = {}
            best_params, _ = grid_optimize(
                base_idea,
                backtest,
                train_bundle,
                ctx,
                space,
                objective,
                constraints,
            )
            result = backtest(best_params, test_bundle, ctx)
        else:
            params: dict = {}
            result = backtest(params, bundle, ctx)
            if constraints and not all(c(result) for c in constraints):
                raise SystemExit("Constraints not satisfied")

        tracker.log_metrics(result.metrics)
        print(json.dumps(result.metrics, indent=2))
    finally:
        tracker.end_run()


if __name__ == "__main__":
    main()
