"""Top-level package for sentimental_cap_predictor.

This module avoids importing heavy optional dependencies at import time by
lazily loading subpackages when they are first accessed.  It keeps the package
importable in lightweight environments used for smoke tests.
"""
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["config", "llm_core", "trading"]


def __getattr__(name: str) -> ModuleType:
    """Dynamically import one of the known subpackages.

    The heavy submodules (for example ``llm_core``) are imported only when they
    are actually accessed, which means simply importing
    :mod:`sentimental_cap_predictor` will not pull in large third‑party
    dependencies such as PyTorch.
    """
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from . import config, llm_core, trading  # noqa: F401
