from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .io_types import Argument, DispatchDecision, NLUResult
from .ontology import Ontology


@dataclass
class Reasoner:
    ontology: Ontology

    def explain(self, decision: DispatchDecision, nlu: NLUResult, ctx: Dict) -> Argument:
        if not decision.executed:
            return Argument(text="No action executed.")
        intent = decision.action or ""
        intent_obj = self.ontology.intents.get(intent)
        utterance = ctx.get("utterance", "").lower()
        signals: List[str] = []
        if intent_obj:
            for sig in intent_obj.signals:
                if sig in utterance:
                    signals.append(sig)
        slot_desc = ", ".join(f"{k}={v}" for k, v in nlu.slots.items()) or "no slots"
        text = (
            f"Action {intent} selected because of signals {signals} and extracted {slot_desc}."
        )
        return Argument(text=text)
