from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class NLUResult:
    """Output of the NLU engine."""

    intent: Optional[str]
    scores: Optional[Dict[str, float]] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)


@dataclass
class Resolution:
    """Resolution from dialog policy."""

    intent: Optional[str]
    action_needed: Literal["ASK_CLARIFY", "ASK_SLOT", "DISPATCH", "FALLBACK"]
    slots: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None


@dataclass
class DispatchDecision:
    """Result of dispatcher invocation."""

    action: Optional[str]
    args: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    result: Any | None = None


@dataclass
class Argument:
    """Human-readable explanation for a dispatch decision."""

    text: str
