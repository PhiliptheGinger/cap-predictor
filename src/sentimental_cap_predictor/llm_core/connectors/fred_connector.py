"""Connector for the Federal Reserve Economic Data (FRED) API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from loguru import logger

FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(
    series_id: str, api_key: str | None = None, **params: Any
) -> List[Dict[str, Any]]:
    """Fetch a FRED time series and return observations as a list of dicts."""

    api_key = api_key or os.environ.get("FRED_API_KEY")
    if api_key is None:
        raise ValueError("FRED API key is required")

    query = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    query.update(params)
    logger.debug("Querying FRED: %s", query)
    response = requests.get(FRED_SERIES_URL, params=query, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("observations", [])


def update_store(
    series_id: str, path: Path, api_key: str | None = None, **params: Any
) -> Path:
    """Fetch a series and persist observations to *path* as JSON."""

    observations = fetch_series(series_id, api_key=api_key, **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observations, indent=2))
    logger.info(
        "Saved %d observations for FRED series %s to %s",
        len(observations),
        series_id,
        path,
    )
    return path


__all__ = ["fetch_series", "update_store"]
