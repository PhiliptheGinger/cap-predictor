"""Tool implementations for :mod:`sentimental_cap_predictor.llm_core.agent`.

Importing this module registers the built-in tools with the agent's global
registry.
"""

# Import tools for side effects so that they register themselves.
from . import web_search  # noqa: F401

__all__ = []
