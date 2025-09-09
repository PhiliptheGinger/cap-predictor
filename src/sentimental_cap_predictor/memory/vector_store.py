from __future__ import annotations

"""Lightweight vector store backed by Pinecone or local Chroma."""

import os
import tempfile
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - graceful degradation
    SentenceTransformer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_EMBEDDER: SentenceTransformer | None = None
_MODEL_FAILED = False
_INDEX: Any | None = None
_USING_PINECONE = False


def _load_model() -> SentenceTransformer | None:
    """Load and cache the sentence-transformer model."""
    global _EMBEDDER, _MODEL_FAILED
    if _EMBEDDER is not None or _MODEL_FAILED:
        return _EMBEDDER
    if SentenceTransformer is None:
        logger.warning("sentence-transformers not installed: %s", _IMPORT_ERROR)
        _MODEL_FAILED = True
        return None
    try:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # pragma: no cover - network failure, etc.
        logger.warning("Unable to load embedding model: %s", exc)
        _MODEL_FAILED = True
        _EMBEDDER = None
    return _EMBEDDER


def _embed(texts: List[str]) -> List[List[float]]:
    model = _load_model()
    if model is None:
        return []
    try:
        return model.encode(texts, convert_to_numpy=True).tolist()
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.warning("Embedding generation failed: %s", exc)
        return []


def ensure_index(name: str = "cap_articles") -> Any | None:
    """Initialize the backing vector store if needed and return the index."""
    global _INDEX, _USING_PINECONE
    if _INDEX is not None:
        return _INDEX

    api_key = os.getenv("PINECONE_API_KEY")
    if api_key:
        try:  # pragma: no cover - optional dependency
            import pinecone
        except Exception as exc:  # pragma: no cover
            logger.warning("Pinecone client unavailable: %s", exc)
        else:
            pinecone.init(api_key=api_key)
            dimension = 384  # embedding size for MiniLM-L6
            try:
                existing = pinecone.list_indexes()  # type: ignore[attr-defined]
                if isinstance(existing, list) and name not in existing:
                    pinecone.create_index(name, dimension=dimension)
            except Exception:  # pragma: no cover - older client versions
                pass
            _INDEX = pinecone.Index(name)
            _USING_PINECONE = True
            return _INDEX

    try:  # pragma: no cover - optional dependency
        import chromadb
    except Exception as exc:  # pragma: no cover
        logger.warning("Chroma client unavailable: %s", exc)
        _INDEX = None
        return None

    persist_dir = tempfile.mkdtemp(prefix="chroma_")
    client = chromadb.PersistentClient(path=persist_dir)
    _INDEX = client.get_or_create_collection(name)
    _USING_PINECONE = False
    return _INDEX


def upsert(doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
    """Add or replace *text* with *doc_id* and *metadata* in the index."""
    index = ensure_index()
    if index is None:
        return
    emb = _embed([text])
    if not emb:
        return
    vector = emb[0]
    if _USING_PINECONE:
        index.upsert([(doc_id, vector, metadata)])  # type: ignore[call-arg]
    else:
        try:
            index.upsert(ids=[doc_id], embeddings=[vector], metadatas=[metadata])
        except AttributeError:
            index.add(ids=[doc_id], embeddings=[vector], metadatas=[metadata])


def query(text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Query the index for *text* and return the top *k* matches."""
    index = ensure_index()
    if index is None:
        return []
    emb = _embed([text])
    if not emb:
        return []
    vector = emb[0]
    if _USING_PINECONE:
        response = index.query(vector=vector, top_k=k, include_metadata=True)
        matches = response.get("matches", []) if isinstance(response, dict) else []
        return [
            {"id": m.get("id"), "score": m.get("score"), "metadata": m.get("metadata", {})}
            for m in matches
        ]
    result = index.query(query_embeddings=[vector], n_results=k, include=["metadatas"])
    ids = result.get("ids", [[]])[0]
    scores = result.get("distances", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    return [
        {"id": i, "score": s, "metadata": m or {}} for i, s, m in zip(ids, scores, metas)
    ]


def available() -> bool:
    """Return ``True`` if the vector DB and embedding model are usable."""

    return ensure_index() is not None and not _MODEL_FAILED


__all__ = ["ensure_index", "upsert", "query", "available"]
