from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback
    import logging as _logging

    logger = _logging.getLogger(__name__)


class RunLogger:
    """Persist details about agent runs to disk.

    Records prompt hash, token usage, tool name, duration and errors.
    Optionally reports to MLflow or Prefect when enabled via
    environment variables ``AGENT_RUN_USE_MLFLOW`` and
    ``AGENT_RUN_USE_PREFECT``.
    """

    def __init__(self, base_dir: str | Path = "reports/agent_runs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._mlflow = None
        self._prefect_logger = None

        if os.getenv("AGENT_RUN_USE_MLFLOW") == "1":  # pragma: no cover - optional
            try:
                import mlflow

                self._mlflow = mlflow
            except Exception:  # pragma: no cover - mlflow not available
                logger.debug("MLflow requested but not available")

        if os.getenv("AGENT_RUN_USE_PREFECT") == "1":  # pragma: no cover - optional
            try:
                from prefect import get_run_logger

                self._prefect_logger = get_run_logger()
            except Exception:  # pragma: no cover - prefect not available
                logger.debug("Prefect requested but not available")

    def log(
        self,
        prompt_hash: str,
        tool_name: str | None,
        tokens_in: int,
        tokens_out: int,
        duration: float,
        error: str | None = None,
    ) -> Path:
        """Write a run record to disk and optional hooks."""

        record: dict[str, Any] = {
            "prompt_hash": prompt_hash,
            "tool_name": tool_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration": duration,
            "error": error,
            "ts": time.time(),
        }
        path = self.base_dir / f"{prompt_hash}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.debug("Logged agent run to {}", path)

        if self._mlflow is not None:  # pragma: no cover - optional
            try:
                self._mlflow.log_dict(record, f"agent_runs/{path.name}")
            except Exception:
                logger.debug("Failed to log to MLflow")

        if self._prefect_logger is not None:  # pragma: no cover - optional
            try:
                self._prefect_logger.info("agent_run %s", record)
            except Exception:
                logger.debug("Failed to log to Prefect")

        return path


__all__ = ["RunLogger"]
