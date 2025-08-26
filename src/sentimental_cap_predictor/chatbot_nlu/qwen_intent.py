"""NLU powered by the Qwen language model.

This module delegates intent classification and slot extraction to an
external Qwen model via a lightweight JSON protocol.  The model is
prompted with a fixed system instruction and the user utterance and is
expected to respond with a JSON object containing ``intent`` and
``slots``.  The result is normalised into :class:`NLUResult` so it can be
consumed by the existing policy/dispatcher pipeline.

The actual network call is isolated in :meth:`_call_model` so tests can
patch it easily without performing real API calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from .io_types import NLUResult
from .ontology import Ontology


SYSTEM_PROMPT = (
    "You are an intent classifier for the Cap Predictor chatbot.\n"
    "Your job is to output JSON only, with \"intent\" and \"slots\".\n"
    "Choose the closest intent from this fixed list: "
    "[pipeline.run_daily, pipeline.run_now, data.ingest, model.train_eval, "
    "plots.make_report, explain.decision, help.show_options]."
)


@dataclass
class QwenNLU:
    """Minimal wrapper around the Qwen model for intent detection."""

    ontology: Ontology
    model: str = "qwen"

    # ------------------------------------------------------------------
    def _call_model(self, utterance: str) -> str:  # pragma: no cover - network
        """Send ``utterance`` to the Qwen model and return its text reply."""

        try:  # pragma: no cover - optional dependency
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": utterance},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message["content"]  # type: ignore[index]
        except Exception as exc:  # pragma: no cover - handled by predict
            raise RuntimeError(f"Qwen call failed: {exc}")

    # ------------------------------------------------------------------
    def predict(self, utterance: str) -> NLUResult:
        """Return :class:`NLUResult` for ``utterance`` using Qwen."""

        try:
            raw = self._call_model(utterance)
            data: Dict[str, Any] = json.loads(raw)
            intent = data.get("intent")
            slots = data.get("slots") or {}
            if intent is None:
                raise ValueError("missing intent")
            # normalise tickers if present
            if "tickers" in slots and isinstance(slots["tickers"], list):
                slots["tickers"] = [str(t).upper() for t in slots["tickers"]]
            if "ticker" in slots and isinstance(slots["ticker"], str):
                slots["ticker"] = slots["ticker"].upper()
            required = self.ontology.required_slots(intent)
            missing = [s for s in required if s not in slots]
            return NLUResult(
                intent=intent, slots=slots, scores={intent: 1.0}, missing_slots=missing
            )
        except Exception:
            return NLUResult(intent="help.show_options", slots={}, scores={})

