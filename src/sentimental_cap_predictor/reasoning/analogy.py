from __future__ import annotations

"""Analogical reasoning helpers."""

from typing import Dict, Tuple


def map_roles(
    source: Dict[str, str],
    target: Dict[str, str],
) -> Dict[str, str]:
    """Align role fillers shared by two conceptual frames.

    Parameters
    ----------
    source:
        Mapping from role names to fillers in the source domain.
    target:
        Mapping from role names to fillers in the target domain.

    Returns
    -------
    Dict[str, str]
        Mapping from each source filler to its counterpart in the target for
        roles that both domains contain. Roles missing in the target are
        ignored.
    """

    mapping: Dict[str, str] = {}
    for role, source_value in source.items():
        if role in target:
            mapping[source_value] = target[role]
    return mapping


# Simple knowledge base of metaphors for demonstration purposes.
_METAPHORS: Dict[str, Tuple[str, str]] = {
    "brain": (
        "computer",
        "Both process information through electrical signals.",
    ),
    "stock market": (
        "rollercoaster",
        "Its values rise and fall dramatically like cars on a ride.",
    ),
    "internet": (
        "highway",
        "Information flows quickly between nodes like vehicles on roads.",
    ),
}


def best_metaphor(target_concept: str) -> Tuple[str, str]:
    """Return a metaphor label and short rationale for a concept.

    Parameters
    ----------
    target_concept:
        Concept to describe metaphorically.

    Returns
    -------
    tuple[str, str]
        ``(label, rationale)`` describing the metaphor. If no metaphor is
        available, the label is empty and the rationale explains this.
    """

    key = target_concept.lower()
    default = ("", f"No metaphor available for {target_concept}")
    return _METAPHORS.get(key, default)
