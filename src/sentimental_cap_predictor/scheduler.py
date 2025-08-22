"""Simple command line scheduler for updating data sources and indexes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import typer
from dotenv import load_dotenv
from loguru import logger

from . import connectors
from .indexer import build_index
from .research.idea_generator import generate_ideas

load_dotenv()

MODEL_HELP = "Model ID to use. Defaults to MAIN_MODEL or 'Qwen/Qwen2-7B-Instruct'."  # noqa: E501

STATE_PATH = Path("state.json")
DATA_DIR = Path("data")

app = typer.Typer(
    add_completion=False,
    help="Data update and indexing scheduler",
)


def _load_state() -> Dict[str, str]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_state(state: Dict[str, str]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


@app.command("update:papers")
def update_papers() -> None:
    """Fetch paper metadata from arXiv and OpenAlex."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    arxiv_path = DATA_DIR / "arxiv_papers.json"
    openalex_path = DATA_DIR / "openalex_papers.json"
    connectors.arxiv_connector.update_store(arxiv_path)
    connectors.openalex_connector.update_store(openalex_path)
    state = _load_state()
    state["papers_updated"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("Paper sources updated")


@app.command("update:pubmed")
def update_pubmed(query: str = "cancer", max_results: int = 10) -> None:
    """Fetch article metadata from PubMed."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pubmed_path = DATA_DIR / "pubmed_papers.json"
    connectors.pubmed_connector.update_store(
        pubmed_path, query=query, max_results=max_results
    )
    state = _load_state()
    state["pubmed_updated"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("PubMed articles updated")


@app.command("update:fred")
def update_fred(series_id: str = "GDP") -> None:
    """Fetch a FRED time series."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fred_path = DATA_DIR / f"fred_{series_id}.json"
    connectors.fred_connector.update_store(series_id, fred_path)
    state = _load_state()
    state[f"fred_{series_id}_updated"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("FRED series %s updated", series_id)


@app.command("update:edgar")
def update_edgar(cik: str) -> None:
    """Fetch SEC EDGAR filings for *cik*."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    edgar_path = DATA_DIR / f"edgar_{cik}.json"
    connectors.edgar_connector.update_store(cik, edgar_path)
    state = _load_state()
    state[f"edgar_{cik}_updated"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("EDGAR filings for %s updated", cik)


@app.command("index:papers")
def index_papers() -> None:
    """Build a FAISS index over all stored papers."""

    arxiv_path = DATA_DIR / "arxiv_papers.json"
    openalex_path = DATA_DIR / "openalex_papers.json"
    pubmed_path = DATA_DIR / "pubmed_papers.json"
    papers: List[dict] = []
    for path in (arxiv_path, openalex_path, pubmed_path):
        if path.exists():
            papers.extend(json.loads(path.read_text()))
    if not papers:
        raise RuntimeError("No paper data available; run update:papers first")

    metadata_path = DATA_DIR / "papers.json"
    metadata_path.write_text(json.dumps(papers, indent=2))
    index_path = DATA_DIR / "papers.index"
    build_index(papers, index_path)

    state = _load_state()
    state["papers_indexed"] = datetime.utcnow().isoformat()
    _save_state(state)
    logger.info("Paper index rebuilt")


@app.command("ideas:generate")
def generate(
    topic: str,
    model_id: str = typer.Option(
        os.getenv("MAIN_MODEL", "Qwen/Qwen2-7B-Instruct"),
        help=MODEL_HELP,
    ),
    n: int = 3,
    output: Path | None = None,
) -> None:
    """Generate trading ideas using a local language model.

    The model ID defaults to the ``MAIN_MODEL`` environment variable or
    ``Qwen/Qwen2-7B-Instruct`` if unset.  The resulting ideas are printed to
    STDOUT or written to ``output`` if provided.  This makes the command easy
    to schedule via cron or other systems.
    """

    ideas = generate_ideas(topic, model_id=model_id, n=n)
    data = [asdict(i) for i in ideas]
    text = json.dumps(data, indent=2)
    if output:
        output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    app()
