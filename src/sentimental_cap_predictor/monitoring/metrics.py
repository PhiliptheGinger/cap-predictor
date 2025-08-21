from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from loguru import logger


def write_kpis(kpis: Dict[str, float], path: str | Path = "reports/kpis.json") -> Path:
    """Write key performance indicators to disk as JSON."""
    path = Path(path)
    path.write_text(json.dumps(kpis, indent=2))
    logger.info("Wrote KPIs to %s", path)
    return path


def check_alerts(
    kpis: Dict[str, float], thresholds: Dict[str, float]
) -> Dict[str, float]:
    """Return KPI values breaching supplied thresholds."""
    alerts = {k: v for k, v in kpis.items() if k in thresholds and v < thresholds[k]}
    if alerts:
        logger.warning("KPI alerts triggered: {}", alerts)
    return alerts
