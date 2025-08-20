from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any, Dict


@dataclass
class ParamSpec:
    """Specification for a parameter in a search space.

    Attributes
    ----------
    kind:
        Type of the parameter. Typical values are ``"int"``, ``"float"`` or
        ``"choice"``.
    bounds:
        Tuple of ``(low, high)`` values for numeric parameters. ``None`` for
        categorical parameters.
    choices:
        Possible values for categorical parameters. ``None`` for numeric
        parameters.
    """

    kind: str
    bounds: tuple[float, float] | None = None
    choices: Sequence[Any] | None = None


# A mapping from parameter names to their specification
SearchSpace = Dict[str, ParamSpec]
