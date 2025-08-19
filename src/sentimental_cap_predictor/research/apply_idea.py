from __future__ import annotations

from typing import Any, Callable

from .idea_schema import Idea


def apply_idea(idea: Idea, func: Callable[[Idea], Any]) -> Any:
    """Apply ``func`` to ``idea`` and return the result."""
    return func(idea)
