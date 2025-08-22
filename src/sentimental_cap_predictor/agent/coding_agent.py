from __future__ import annotations

"""Placeholder coding agent utilities.

This module sketches out interfaces for generating and applying code changes.
The implementations are deliberately minimal and do not modify the repository.
"""

from .dispatcher import DispatchResult


def propose_changes(objective: str, context: str) -> str:
    """Return a placeholder patch for ``objective`` given ``context``.

    Parameters
    ----------
    objective:
        High level description of the requested change.
    context:
        Supplementary information or source content to consider.
    """

    return (
        "---\n"
        f"# objective: {objective}\n"
        f"# context: {context}\n"
        "+++\n"
    )


def apply_changes(patch: str) -> DispatchResult:
    """Stub handler that pretends to apply ``patch`` to the codebase.

    The operation is intentionally disabled. The function merely returns a
    :class:`DispatchResult` communicating that the command is unavailable.
    """

    return DispatchResult(ok=False, message="code.implement disabled")
