"""Vector store memory operations for agent tools."""

from __future__ import annotations

from typing import Any, Dict, List

from sentimental_cap_predictor.memory import vector_store


def memory_upsert(id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
    """Insert or update ``text`` with ``id`` and ``metadata`` in the vector store."""
    vector_store.upsert(id, text, metadata or {})


def memory_query(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return up to ``top_k`` matches for ``query`` from the vector store."""
    return vector_store.query(query, k=top_k)


__all__ = ["memory_upsert", "memory_query"]

# Optional agent tool registration
try:  # pragma: no cover - registration is optional at runtime
    from pydantic import BaseModel, Field

    from sentimental_cap_predictor.llm_core.agent.tool_registry import (
        ToolSpec,
        register_tool,
    )

    class MemoryUpsertInput(BaseModel):
        id: str
        text: str
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class MemoryUpsertOutput(BaseModel):
        success: bool

    def _memory_upsert_handler(payload: MemoryUpsertInput) -> MemoryUpsertOutput:
        memory_upsert(payload.id, payload.text, payload.metadata)
        return MemoryUpsertOutput(success=True)

    register_tool(
        ToolSpec(
            name="memory.upsert",
            input_model=MemoryUpsertInput,
            output_model=MemoryUpsertOutput,
            handler=_memory_upsert_handler,
        )
    )

    class MemoryQueryInput(BaseModel):
        query: str
        top_k: int = 5

    class MemoryQueryOutput(BaseModel):
        results: List[Dict[str, Any]]

    def _memory_query_handler(payload: MemoryQueryInput) -> MemoryQueryOutput:
        hits = memory_query(payload.query, payload.top_k)
        return MemoryQueryOutput(results=hits)

    register_tool(
        ToolSpec(
            name="memory.query",
            input_model=MemoryQueryInput,
            output_model=MemoryQueryOutput,
            handler=_memory_query_handler,
        )
    )
except Exception:  # pragma: no cover - silently ignore registration issues
    pass
