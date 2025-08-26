from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .io_types import NLUResult, Resolution
from .ontology import Ontology

INTENT_THRESHOLD = 0.72
AMBIG_MARGIN = 0.08


@dataclass
class Policy:
    ontology: Ontology

    def resolve(self, nlu: NLUResult, ctx: Dict) -> Resolution:
        scores = nlu.scores or {}
        if not scores and nlu.intent:
            scores = {nlu.intent: 1.0}
        if not scores:
            return Resolution(intent=None, slots={}, action_needed="FALLBACK", prompt="I didn't catch that.")
        intents_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = intents_sorted[0]
        second_score = intents_sorted[1][1] if len(intents_sorted) > 1 else 0.0

        # if top2 scores are close trigger clarification
        if top_score - second_score < AMBIG_MARGIN:
            prompt = f"Did you mean {top_intent} or {intents_sorted[1][0]}?"
            return Resolution(intent=None, action_needed="ASK_CLARIFY", slots={}, prompt=prompt)

        # unknown or help intent -> fallback directly
        if top_intent == "help.show_options":
            return Resolution(
                intent="help.show_options",
                slots={},
                action_needed="FALLBACK",
                prompt="I can run pipelines, ingest data, train models, or plot reports.",
            )

        if top_score < INTENT_THRESHOLD:
            return Resolution(
                intent="help.show_options",
                slots={},
                action_needed="FALLBACK",
                prompt="I'm not sure what you need. Try 'help' for options.",
            )

        if nlu.missing_slots:
            prompt = "Please provide: " + ", ".join(nlu.missing_slots)
            return Resolution(intent=top_intent, slots=nlu.slots, action_needed="ASK_SLOT", prompt=prompt)

        return Resolution(intent=top_intent, slots=nlu.slots, action_needed="DISPATCH", prompt=None)
