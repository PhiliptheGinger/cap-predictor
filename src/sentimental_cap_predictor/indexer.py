"""Utilities for building and querying a FAISS index over papers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import faiss  # type: ignore
import torch
from loguru import logger

if TYPE_CHECKING:
    import numpy as np

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model(
    model_name: str = MODEL_NAME,
) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_texts(
    texts: Sequence[str],
    model_name: str = MODEL_NAME,
) -> "np.ndarray":
    """Return ``float32`` sentence embeddings for *texts*."""

    tokenizer, model = _load_model(model_name)
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")  # noqa: E501
    return embeddings


def build_index(
    papers: Sequence[dict],
    index_path: Path,
    model_name: str = MODEL_NAME,
) -> Path:
    """Build a FAISS index for *papers* and persist it to *index_path*.

    The papers are expected to contain ``title`` and ``abstract`` fields.  The
    embeddings are constructed from the concatenation of these fields.
    Returns the path to the written index.
    """

    texts: List[str] = []
    for p in papers:
        text = f"{p.get('title', '')} {p.get('abstract', '')}".strip()
        texts.append(text)
    if not texts:
        raise ValueError("No papers supplied")
    embeddings = embed_texts(texts, model_name=model_name)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(
        "Wrote FAISS index with %d vectors to %s",
        len(texts),
        index_path,
    )
    return index_path


def load_papers(path: Path) -> List[dict]:
    """Load papers metadata from *path* (JSON list)."""

    return json.loads(path.read_text())


__all__ = ["embed_texts", "build_index", "load_papers"]
