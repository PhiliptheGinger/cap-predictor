"""Basic file writing utilities for agent operations."""

from __future__ import annotations

import logging
from pathlib import Path

MAX_WRITE_BYTES = 1_000_000
"""Maximum size of ``content`` allowed for :func:`file_write` in bytes."""


def file_write(path: str, content: str) -> str:
    """Write ``content`` to ``agent_work`` directory under the given ``path``.

    The function ensures that the resolved target path resides within the
    ``agent_work`` directory. If the target file already exists a warning is
    logged prior to overwriting. A simple size check prevents writing content
    larger than :data:`MAX_WRITE_BYTES`.

    Parameters
    ----------
    path:
        Relative path below ``agent_work`` where the file should be written.
    content:
        Textual content to persist.

    Returns
    -------
    str
        The absolute path of the written file.

    Raises
    ------
    ValueError
        If the resolved path escapes ``agent_work`` or the content exceeds the
        maximum allowed size.
    """

    base = Path("agent_work").resolve()
    target = (base / path).resolve()

    if not str(target).startswith(str(base)):
        raise ValueError("path must reside within 'agent_work/'")

    size = len(content.encode("utf-8"))
    if size > MAX_WRITE_BYTES:
        raise ValueError("content exceeds maximum allowed size")

    if target.exists():
        logging.warning("Overwriting existing file at %s", target)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return str(target)


__all__ = ["file_write"]

# Optional agent tool registration
try:  # pragma: no cover - registration is optional at runtime
    from pydantic import BaseModel

    from sentimental_cap_predictor.llm_core.agent.tool_registry import (
        ToolSpec,
        register_tool,
    )

    class FileWriteInput(BaseModel):
        path: str
        content: str

    class FileWriteOutput(BaseModel):
        path: str

    def _file_write_handler(payload: FileWriteInput) -> FileWriteOutput:
        written_path = file_write(payload.path, payload.content)
        return FileWriteOutput(path=written_path)

    register_tool(
        ToolSpec(
            name="file.write",
            input_model=FileWriteInput,
            output_model=FileWriteOutput,
            handler=_file_write_handler,
        )
    )
except Exception:  # pragma: no cover - silently ignore registration issues
    pass
