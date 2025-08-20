from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, index=True)
    key = Column(String)
    value = Column(Float)
    step = Column(Integer, nullable=True)


class LocalTracker:
    """A simple SQLite-backed experiment tracker."""

    def __init__(self, root: str = "./runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.root / 'runs.sqlite'}")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._current_run: Optional[str] = None

    def start_run(self) -> str:
        run_id = uuid.uuid4().hex
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with self.SessionLocal() as session:
            session.add(Run(run_id=run_id, start_time=datetime.utcnow()))
            session.commit()
        self._current_run = run_id
        return run_id

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        with self.SessionLocal() as session:
            for key, value in metrics.items():
                session.add(
                    Metric(
                        run_id=self._current_run,
                        key=key,
                        value=float(value),
                        step=step,
                    )
                )
            session.commit()

    def log_artifact(self, path: str | Path) -> None:
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(path)
        dest = self.root / self._current_run / src.name
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)

    def end_run(self) -> None:
        if self._current_run is None:
            return
        with self.SessionLocal() as session:
            run = session.query(Run).filter_by(run_id=self._current_run).one()
            run.end_time = datetime.utcnow()
            session.commit()
        self._current_run = None


class MlflowTracker:
    """Thin wrapper around MLflow's tracking API."""

    def __init__(self) -> None:
        import mlflow

        self.mlflow = mlflow

    def start_run(self) -> Any:  # pragma: no cover - simple wrapper
        return self.mlflow.start_run()

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self.mlflow.log_artifact(str(path))

    def end_run(self) -> None:
        self.mlflow.end_run()


def get_tracker() -> LocalTracker | MlflowTracker:
    """Return the appropriate tracker based on environment configuration."""

    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            return MlflowTracker()
        except Exception:
            pass
    return LocalTracker()
