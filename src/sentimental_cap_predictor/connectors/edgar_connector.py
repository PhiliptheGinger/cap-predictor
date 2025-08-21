"""Connector for SEC EDGAR filings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
from loguru import logger

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def fetch_filings(cik: str) -> List[Dict[str, Any]]:
    """Fetch recent filings for a given *cik* from the SEC EDGAR API."""

    cik = cik.zfill(10)
    headers = {"User-Agent": "cap-predictor/0.1"}
    logger.debug("Querying EDGAR for CIK %s", cik)
    response = requests.get(
        EDGAR_SUBMISSIONS_URL.format(cik=cik), headers=headers, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    recent = data.get("filings", {}).get("recent", {})
    count = len(recent.get("accessionNumber", []))
    filings: List[Dict[str, Any]] = []
    for i in range(count):
        filing = {key: recent[key][i] for key in recent}
        filings.append(filing)
    return filings


def update_store(cik: str, path: Path) -> Path:
    """Fetch filings and persist them to *path* as JSON."""

    filings = fetch_filings(cik)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(filings, indent=2))
    logger.info("Saved %d filings for CIK %s to %s", len(filings), cik, path)
    return path


__all__ = ["fetch_filings", "update_store"]
