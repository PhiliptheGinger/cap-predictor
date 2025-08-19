from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Idea:
    """Representation of a research idea.

    Attributes
    ----------
    name:
        Human readable name of the idea.
    description:
        Short summary of the hypothesis being tested.
    params:
        Arbitrary parameter dictionary used by downstream experiments.
    """

    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
