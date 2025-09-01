"""Simple idea mining utilities built on top of the paper index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss  # type: ignore
from loguru import logger

from .memory_indexer import MODEL_NAME, TextMemory

INDEX_PATH = Path("data/papers.index")
METADATA_PATH = Path("data/papers.json")


def retrieve_context(query: str, k: int = 5) -> List[dict]:
    """Return *k* papers most relevant to *query*.

    The function expects the FAISS index and metadata JSON to have been
    produced beforehand.  A list of paper dictionaries is returned ordered
    by relevance.
    """

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Index or metadata not found; run indexer first",
        )

    logger.debug("Loading FAISS index from %s", INDEX_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    papers = json.loads(METADATA_PATH.read_text())

    memory = TextMemory(model_name=MODEL_NAME)
    query_vec = memory.embed([query])
    distances, indices = index.search(query_vec, k)

    results: List[dict] = []
    for idx in indices[0]:
        if int(idx) < len(papers):
            results.append(papers[int(idx)])
    return results


__all__ = ["retrieve_context"]
