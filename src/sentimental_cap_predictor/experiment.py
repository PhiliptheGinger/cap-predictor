from __future__ import annotations

import hashlib
import json
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

        Returns the database row id for the newly created run so that additional
        artifacts can be added later.
        """

        code_hash = hashlib.sha256(code.encode()).hexdigest()
        cursor = self.conn.execute(
            "INSERT INTO runs(code_hash, params, metrics, artifacts) VALUES (?, ?, ?, ?)",
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
            "SELECT id, code_hash, params, metrics, artifacts, timestamp FROM runs ORDER BY id DESC"
        )
        return [self._deserialize(row) for row in cursor.fetchall()]

    def get_run(self, run_id: int) -> Dict[str, Any]:
        """Retrieve a single run by id."""

        cursor = self.conn.execute(
            "SELECT id, code_hash, params, metrics, artifacts, timestamp FROM runs WHERE id=?",
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
            raise KeyError(f"Artifact {name!r} not found for run {run_id}") from exc

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
# CLI helpers using Typer
# ----------------------------------------------------------------------
app = typer.Typer(help="Minimal CLI for inspecting experiment runs")


@app.command("list")
def cli_list_runs() -> None:
    """List all runs with their metrics."""

    tracker = ExperimentTracker()
    for run in tracker.list_runs():
        typer.echo(f"{run['id']}: params={run['params']} metrics={run['metrics']}")


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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
