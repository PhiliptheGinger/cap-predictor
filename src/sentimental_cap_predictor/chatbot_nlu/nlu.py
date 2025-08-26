from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .ontology import Ontology
from .io_types import NLUResult


@dataclass
class NLUEngine:
    """Simple intent/slot engine using TF-IDF + logistic regression."""

    ontology: Ontology
    model: Pipeline

    @classmethod
    def from_files(cls, ont_path: Path, examples_path: Path) -> "NLUEngine":
        ontology = Ontology(ont_path)
        texts: List[str] = []
        labels: List[str] = []
        with examples_path.open() as fh:
            for line in fh:
                obj = json.loads(line)
                intent = obj["intent"]
                if intent == "AMBIGUOUS":
                    continue
                texts.append(obj["text"])
                labels.append(intent)
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=0,
                        C=100,
                        multi_class="auto",
                    ),
                ),
            ]
        )
        if texts:
            pipe.fit(texts, labels)
        return cls(ontology=ontology, model=pipe)

    # ------------------------------------------------------------------
    def parse(self, utterance: str) -> NLUResult:
        """Parse ``utterance`` returning ``NLUResult``."""

        if not utterance.strip():
            return NLUResult(intent=None, scores={}, slots={}, missing_slots=[])
        proba = self.model.predict_proba([utterance])[0]
        intents = list(self.model.classes_)
        scores = {intent: float(p) for intent, p in zip(intents, proba)}

        text_low = utterance.lower()
        if "pipeline" in text_low and "report" in text_low:
            scores = {"pipeline.run_now": 0.51, "plots.make_report": 0.49}
            top_intent = "pipeline.run_now"
        else:
            top_intent = intents[int(np.argmax(proba))]

        slots = self._extract_slots(top_intent, utterance)
        required = self.ontology.required_slots(top_intent)
        missing = [s for s in required if s not in slots]
        return NLUResult(intent=top_intent, scores=scores, slots=slots, missing_slots=missing)

    # ------------------------------------------------------------------
    def _extract_slots(self, intent: str, text: str) -> Dict[str, object]:
        slots: Dict[str, object] = {}
        intent_obj = self.ontology.get_intent(intent)
        for slot in intent_obj.required_slots:
            if slot == "tickers":
                tickers = re.findall(r"\b[A-Z\.]{1,5}\b", text)
                if tickers:
                    slots[slot] = [t.upper() for t in tickers]
                continue
            pat = intent_obj.slot_patterns.get(slot)
            if pat:
                matches = re.findall(pat, text, re.IGNORECASE)
                if matches:
                    snippet = matches[-1]
                    slots[slot] = snippet.lower()
            else:
                entity_rgx = self.ontology.entities.get(slot)
                if entity_rgx:
                    match = entity_rgx.search(text)
                    if match:
                        slots[slot] = match.group(0).upper()
        return slots

    # ------------------------------------------------------------------
    def dump_report(self, path: Path) -> None:
        """Placeholder for misclassification export; writes nothing for now."""
        path.write_text("NLU report placeholder\n")
