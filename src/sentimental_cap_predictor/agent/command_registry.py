from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
import platform
import sys

import pytest

from sentimental_cap_predictor import experiment, plots
from sentimental_cap_predictor.data import ingest as data_ingest
from sentimental_cap_predictor.flows import daily_pipeline
from sentimental_cap_predictor.modeling import train_eval as model_train_eval
from sentimental_cap_predictor.research import idea_generator
from sentimental_cap_predictor.trader_utils import strategy_optimizer


@dataclass
class Command:
    """Definition of an executable command."""

    name: str
    handler: Callable[..., Any]
    summary: str
    params_schema: Mapping[str, Any] | None = None
    dangerous: bool = False
    aliases: Sequence[str] = field(default_factory=tuple)


def run_tests(args: Sequence[str] | None = None) -> int:
    """Execute the project's pytest suite."""

    return pytest.main(list(args) if args else [])


def read_file(path: str) -> str:
    """Return the text contents of ``path``."""

    return Path(path).read_text()


def system_status() -> Dict[str, str]:
    """Return basic information about the Python runtime and platform."""

    return {"python": sys.version, "platform": platform.platform()}




def promote_model(src: str, dst: str, dry_run: bool | None = False) -> Dict[str, Any]:
    """Swap model config and weights between ``src`` and ``dst`` directories."""

    def _find(directory: Path, stem: str) -> Path:
        matches = list(directory.glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"missing {stem} in {directory}")
        if len(matches) > 1:
            raise ValueError(f"multiple {stem} files in {directory}")
        return matches[0]

    dry = bool(dry_run)
    src_dir = Path(src)
    dst_dir = Path(dst)
    pairs = [
        (_find(src_dir, "config"), _find(dst_dir, "config")),
        (_find(src_dir, "weights"), _find(dst_dir, "weights")),
    ]
    artifacts = [str(pairs[0][1]), str(pairs[1][1])]
    if dry:
        return {
            "summary": f"would swap {src} -> {dst}",
            "artifacts": artifacts,
        }
    for a, b in pairs:
        tmp = b.with_suffix(b.suffix + ".tmp")
        b.rename(tmp)
        a.rename(b)
        tmp.rename(a)
    return {"summary": f"promoted {src} -> {dst}", "artifacts": artifacts}


def get_registry() -> Dict[str, Command]:
    """Return mapping of command names to :class:`Command` entries."""
    from sentimental_cap_predictor.agent import coding_agent
    from .sandbox import safe_shell

    from .sandbox import safe_shell

    return {
        "data.ingest": Command(
            name="data.ingest",
            handler=data_ingest.main,
            summary="Download and prepare price data",
            params_schema={
                "ticker": "str",
                "period": "str",
                "interval": "str",
                "offline_path": "Path|None",
            },
        ),
        "model.train_eval": Command(
            name="model.train_eval",
            handler=model_train_eval.main,
            summary="Train baseline models and write evaluation CSVs",
            params_schema={"ticker": "str"},
        ),
        "plots.generate": Command(
            name="plots.generate",
            handler=plots.main,
            summary="Generate prediction and learning curve plots",
            params_schema={
                "ticker_symbol": "str",
                "mode": "str",
                "output_path": "Path|None",
                "num_lines": "int",
            },
        ),
        "pipeline.run_daily": Command(
            name="pipeline.run_daily",
            handler=daily_pipeline.run,
            summary="Run full daily pipeline",
            params_schema={
                "ticker": "str",
                "period": "str",
                "interval": "str",
            },
            dangerous=True,
            aliases=(
                "run the daily pipeline",
                "run the full pipeline",
                "run the entire pipeline",
                "run the whole pipeline",
            ),
        ),
        "strategy.optimize": Command(
            name="strategy.optimize",
            handler=strategy_optimizer.optimize,
            summary="Optimize strategy parameters via random search",
            params_schema={
                "csv_path": "str",
                "iterations": "int",
                "seed": "int|None",
                "lambda_drawdown": "float",
            },
        ),
        "ideas.generate": Command(
            name="ideas.generate",
            handler=idea_generator.generate_ideas,
            summary="Generate trading ideas using a local model",
            params_schema={"topic": "str", "model_id": "str", "n": "int"},
        ),
        "experiments.list": Command(
            name="experiments.list",
            handler=experiment.list_runs,
            summary="List recorded experiment runs",
            params_schema={},
        ),
        "experiments.show": Command(
            name="experiments.show",
            handler=experiment.show_run,
            summary="Show details for a single experiment run",
            params_schema={"run_id": "int"},
        ),
        "experiments.compare": Command(
            name="experiments.compare",
            handler=experiment.compare_runs,
            summary="Compare metrics of two experiment runs",
            params_schema={"first": "int", "second": "int"},
        ),
        "model.promote": Command(
            name="model.promote",
            handler=promote_model,
            summary="Promote model artifact to production",
            params_schema={"src": "str", "dst": "str", "dry_run": "bool|None"},
            dangerous=True,
        ),
        "tests.run": Command(
            name="tests.run",
            handler=run_tests,
            summary="Run the project's test suite",
            params_schema={"args": "Sequence[str]|None"},
        ),
        "file.read": Command(
            name="file.read",
            handler=read_file,
            summary="Read a text file from disk",
            params_schema={"path": "str"},
        ),
        "sys.status": Command(
            name="sys.status",
            handler=system_status,
            summary="Report basic system information",
            params_schema={},
        ),
        "code.implement": Command(
            name="code.implement",
            handler=coding_agent.apply_changes,
            summary="Apply a code patch generated elsewhere",
            params_schema={"patch": "str"},
            dangerous=True,
        ),
        "shell.run": Command(
            name="shell.run",
            handler=safe_shell,
            summary="Execute a shell command in a restricted sandbox",
            params_schema={"cmd": "str"},
            dangerous=True,
        ),
    }
