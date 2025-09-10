from __future__ import annotations

"""High level reasoning engine built atop simple helpers."""

from typing import Any, Dict, List, Optional

from ..memory import vector_store
from .analogy import best_metaphor, map_roles
from .simulator import step as sim_step


def reason_about(
    text: str,
    memory_hits: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Return an explanation contrasting retrieved facts and speculation.

    Parameters
    ----------
    text:
        Claim or question to analyze.
    memory_hits:
        Optional list of matches from :mod:`vector_store`. If ``None``, the
        store is queried directly when available.

    Returns
    -------
    str
        Multi-line explanation where factual snippets precede speculative
        reasoning.
    """

    hits = memory_hits
    queried = False
    if hits is None:
        if vector_store.available():
            hits = vector_store.query(text)
            queried = True
        else:
            hits = []

    facts: List[str] = []
    if queried and not hits:
        facts.append("- No relevant memories found.")

    for hit in hits or []:
        meta = hit.get("metadata", {}) or {}
        snippet = meta.get("text")
        if not snippet:
            snippet = meta.get("snippet")
        if not snippet:
            snippet = meta.get("summary")
        source = meta.get("source") or hit.get("id", "unknown")
        if snippet:
            facts.append(f"- {snippet} (source: {source})")

    if facts:
        fact_block = "Facts:\n" + "\n".join(facts)
    else:
        fact_block = "Facts:\n- No supporting evidence retrieved."

    spec_text = text if text else "No speculation provided."
    spec_block = f"Speculation:\n{spec_text}"
    return f"{fact_block}\n\n{spec_block}"


def analogy_explain(src: str, tgt: str) -> str:
    """Provide a short analogy comparing ``src`` and ``tgt`` with caveats."""

    src_name = src if isinstance(src, str) else str(src.get("name", "source"))
    tgt_name = tgt if isinstance(tgt, str) else str(tgt.get("name", "target"))

    mapping_text = ""
    if isinstance(src, dict) and isinstance(tgt, dict):
        role_mapping = map_roles(src, tgt)
        if role_mapping:
            pairs = ", ".join(f"{s}->{t}" for s, t in role_mapping.items())
            mapping_text = f" Shared roles: {pairs}."
        else:
            mapping_text = " The two domains have no analogous roles."

    label, rationale = best_metaphor(tgt_name)
    if label:
        para1 = (
            f"{tgt_name.capitalize()} can be likened to a {label}. {rationale} "
            f"This mirrors aspects of {src_name}."
        )
    else:
        para1 = (
            f"There is no ready-made metaphor for {tgt_name}. "
            f"Still, comparing it to {src_name} can offer intuition."
        )

    para2 = (
        f"However, {tgt_name} and {src_name} are not identical, "
        "so the analogy should be taken as a loose comparison "
        "rather than a strict equivalence."
    )
    return f"{para1}{mapping_text}\n\n{para2}"


def simulate(scenario: str) -> str:
    """Parse qualitative ``scenario`` and narrate one step of the simulator."""

    text = scenario.lower()
    valid_trends = {"up", "down"}
    valid_forces = {"strong", "weak"}

    trend = next((t for t in valid_trends if t in text), None)
    force = next((f for f in valid_forces if f in text), None)

    if trend is None or force is None:
        return "Generic outcome: no specific trend or force recognized."

    state: Dict[str, str] = {"trend": trend, "force": force}
    try:
        next_state, narration = sim_step(state)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"Simulation failed: {exc}"

    return (
        f"Starting with trend={trend} and force={force}. "
        f"{narration} Next state: trend={next_state['trend']}, "
        f"force={next_state['force']}."
    )
