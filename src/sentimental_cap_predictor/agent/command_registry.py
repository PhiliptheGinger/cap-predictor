from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
import platform
import shutil
import subprocess
import sys

import pytest

from sentimental_cap_predictor.data import ingest as data_ingest
from sentimental_cap_predictor.modeling import train_eval as model_train_eval
from sentimental_cap_predictor import plots
from sentimental_cap_predictor.flows import daily_pipeline
from sentimental_cap_predictor.trader_utils import strategy_optimizer
from sentimental_cap_predictor.research import idea_generator
from sentimental_cap_predictor import experiment


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


def run_shell(cmd: str) -> subprocess.CompletedProcess[str]:
    """Execute ``cmd`` in the system shell."""

    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def promote_model(src: str, dst: str) -> str:
    """Copy a model artifact from ``src`` to ``dst``."""

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)
    return str(dst_path)


def get_registry() -> Dict[str, Command]:
    """Return mapping of command names to :class:`Command` entries."""

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
            params_schema={"ticker": "str", "period": "str", "interval": "str"},
            dangerous=True,
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
            handler=experiment.cli_list_runs,
            summary="List recorded experiment runs",
            params_schema={},
        ),
        "experiments.show": Command(
            name="experiments.show",
            handler=experiment.show,
            summary="Show details for a single experiment run",
            params_schema={"run_id": "int"},
        ),
        "experiments.compare": Command(
            name="experiments.compare",
            handler=experiment.compare,
            summary="Compare metrics of two experiment runs",
            params_schema={"first": "int", "second": "int"},
        ),
        "model.promote": Command(
            name="model.promote",
            handler=promote_model,
            summary="Promote model artifact to production",
            params_schema={"src": "str", "dst": "str"},
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
        "shell.run": Command(
            name="shell.run",
            handler=run_shell,
            summary="Execute a shell command",
            params_schema={"cmd": "str"},
            dangerous=True,
        ),
    }
