from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import typer
from loguru import logger

DB_PATH = Path("experiments.db")


class ExperimentTracker:
    """Very small SQLite-based experiment tracker."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT,
                params TEXT,
                metrics TEXT,
                artifacts TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def log(
        self,
        code: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Dict[str, str],
    ) -> int:
        """Log a new experiment run.

        Returns the database row id for the newly created run so that
        additional artifacts can be added later.
        """

        code_hash = hashlib.sha256(code.encode()).hexdigest()
        cursor = self.conn.execute(
            (
                "INSERT INTO runs(code_hash, params, metrics, artifacts) "
                "VALUES (?, ?, ?, ?)"
            ),
            (
                code_hash,
                json.dumps(params, default=str),
                json.dumps(metrics, default=str),
                json.dumps(artifacts, default=str),
            ),
        )
        self.conn.commit()
        run_id = cursor.lastrowid
        logger.info("Logged experiment %s as run %s", code_hash, run_id)
        return int(run_id)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _deserialize(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row into a dictionary."""

        return {
            "id": row[0],
            "code_hash": row[1],
            "params": json.loads(row[2]) if row[2] else {},
            "metrics": json.loads(row[3]) if row[3] else {},
            "artifacts": json.loads(row[4]) if row[4] else {},
            "timestamp": row[5],
        }

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all recorded experiment runs."""

        cursor = self.conn.execute(
            (
                "SELECT id, code_hash, params, metrics, artifacts, timestamp "
                "FROM runs ORDER BY id DESC"
            )
        )
        return [self._deserialize(row) for row in cursor.fetchall()]

    def get_run(self, run_id: int) -> Dict[str, Any]:
        """Retrieve a single run by id."""

        cursor = self.conn.execute(
            (
                "SELECT id, code_hash, params, metrics, artifacts, timestamp "
                "FROM runs WHERE id=?"
            ),
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Run {run_id} not found")
        return self._deserialize(row)

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------
    def add_artifact(self, run_id: int, name: str, path: str | Path) -> None:
        """Add or update an artifact path for an existing run."""

        run = self.get_run(run_id)
        artifacts = run["artifacts"]
        artifacts[name] = str(path)
        self.conn.execute(
            "UPDATE runs SET artifacts=? WHERE id=?",
            (json.dumps(artifacts, default=str), run_id),
        )
        self.conn.commit()

    def get_artifacts(self, run_id: int) -> Dict[str, Path]:
        """Return a mapping of artifact names to ``Path`` objects."""

        run = self.get_run(run_id)
        return {name: Path(p) for name, p in run["artifacts"].items()}

    def get_artifact_path(self, run_id: int, name: str) -> Path:
        """Get the ``Path`` for a specific artifact."""

        artifacts = self.get_artifacts(run_id)
        try:
            return artifacts[name]
        except KeyError as exc:
            msg = f"Artifact {name!r} not found for run {run_id}"
            raise KeyError(msg) from exc

    def load_artifact(self, run_id: int, name: str, mode: str = "r") -> Any:
        """Load the contents of an artifact.

        Parameters
        ----------
        run_id: int
            Identifier of the run.
        name: str
            Artifact name.
        mode: str
            File open mode. "r" for text, "rb" for binary. Defaults to text.
        """

        path = self.get_artifact_path(run_id, name)
        return path.read_text() if "b" not in mode else path.read_bytes()


# ----------------------------------------------------------------------
# Structured helpers
# ----------------------------------------------------------------------


def list_runs() -> Dict[str, Any]:
    """Return summary of all runs with their metrics and artifacts."""

    tracker = ExperimentTracker()
    runs = tracker.list_runs()
    metrics: Dict[str, Any] = {str(r["id"]): r["metrics"] for r in runs}
    artifacts = [str(p) for r in runs for p in r["artifacts"].values()]
    return {
        "summary": f"{len(runs)} runs",
        "metrics": metrics,
        "artifacts": artifacts,
    }


def show_run(run_id: int) -> Dict[str, Any]:
    """Return metrics and artifacts for a single run."""

    tracker = ExperimentTracker()
    run = tracker.get_run(run_id)
    return {
        "summary": f"Run {run_id}",
        "metrics": run["metrics"],
        "artifacts": list(run["artifacts"].values()),
    }


def compare_runs(first: int, second: int) -> Dict[str, Any]:
    """Compare metrics of two runs and surface their artifacts."""

    tracker = ExperimentTracker()
    run_a = tracker.get_run(first)
    run_b = tracker.get_run(second)
    shared = set(run_a["metrics"]).intersection(run_b["metrics"])
    diff = {
        key: run_a["metrics"].get(key, 0) - run_b["metrics"].get(key, 0)
        for key in shared
    }
    artifacts = list(run_a["artifacts"].values()) + list(
        run_b["artifacts"].values()
    )
    return {
        "summary": f"{first} vs {second}",
        "metrics": {
            "first": run_a["metrics"],
            "second": run_b["metrics"],
            "diff": diff,
        },
        "artifacts": artifacts,
    }


# ----------------------------------------------------------------------
# CLI helpers using Typer
# ----------------------------------------------------------------------
app = typer.Typer(help="Minimal CLI for inspecting experiment runs")


@app.command("list")
def cli_list_runs() -> None:
    """List all runs with their metrics."""

    tracker = ExperimentTracker()
    for run in tracker.list_runs():
        msg = f"{run['id']}: params={run['params']} metrics={run['metrics']}"
        typer.echo(msg)


@app.command()
def show(run_id: int) -> None:
    """Show details for a single run."""

    tracker = ExperimentTracker()
    run = tracker.get_run(run_id)
    typer.echo(json.dumps(run, indent=2))


@app.command()
def compare(first: int, second: int) -> None:
    """Compare metrics of two runs."""

    tracker = ExperimentTracker()
    run_a = tracker.get_run(first)
    run_b = tracker.get_run(second)
    typer.echo(f"Comparing run {first} vs {second}")
    for key in set(run_a["metrics"]).union(run_b["metrics"]):
        a = run_a["metrics"].get(key)
        b = run_b["metrics"].get(key)
        typer.echo(f"{key}: {a} vs {b}")


def _compare_explanation(
    tracker: ExperimentTracker, first: int, second: int, metric: str
) -> str:
    """Generate a textual comparison for two runs on a metric.

    The explanation cites the metric values so the reasoning can be audited.
    """

    run_a = tracker.get_run(first)
    run_b = tracker.get_run(second)
    a = run_a["metrics"].get(metric)
    b = run_b["metrics"].get(metric)
    if a is None or b is None:
        return f"Metric {metric!r} missing for one of the runs."
    if a > b:
        winner = first
    elif b > a:
        winner = second
    else:
        return (
            f"Runs {first} and {second} tie on {metric}: "
            f"{a} vs {b}. Source: experiment tracker"
        )
    loser = second if winner == first else first
    return (
        f"Run {winner} favored over {loser} on {metric}: {a} vs {b}. "
        f"Source: experiment tracker"
    )


def _handle_question(message: str, tracker: ExperimentTracker) -> str:
    """Very small chatbot brain for answering comparison questions."""

    match = re.search(r"compare\s+(\d+)\s+(\d+)\s+metric=(\w+)", message, re.I)
    if not match:
        msg = "I can compare runs. Ask: 'compare <id1> <id2> metric=<metric>'."
        return msg
    first, second, metric = match.groups()
    return _compare_explanation(tracker, int(first), int(second), metric)


@app.command()
def chat() -> None:
    """Interactive chatbot for explaining experiment comparisons."""

    tracker = ExperimentTracker()
    typer.echo(
        "Experiment chatbot ready. Type 'exit' to quit.\n"
        "Ask: compare <id1> <id2> metric=<metric>"
    )
    while True:
        question = typer.prompt("You")
        if question.strip().lower() in {"exit", "quit"}:
            break
        answer = _handle_question(question, tracker)
        typer.echo(f"Bot: {answer}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
