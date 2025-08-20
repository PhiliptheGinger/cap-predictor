from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any

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

    def log(self, code: str, params: Dict[str, Any], metrics: Dict[str, float], artifacts: Dict[str, str]) -> None:
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.conn.execute(
            "INSERT INTO runs(code_hash, params, metrics, artifacts) VALUES (?, ?, ?, ?)",
            (
                code_hash,
                json.dumps(params, default=str),
                json.dumps(metrics, default=str),
                json.dumps(artifacts, default=str),
            ),
        )
        self.conn.commit()
        logger.info("Logged experiment with hash %s", code_hash)
