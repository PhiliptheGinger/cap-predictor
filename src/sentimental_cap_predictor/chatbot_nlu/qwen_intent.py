"""NLU engine backed by the Qwen model.

The class sends a prompt to a Qwen endpoint which performs both intent
classification and slot extraction.  Only a small fixed set of intents is
supported.  The model is expected to reply with a JSON object containing the
predicted ``intent`` and optional ``slots`` mapping.

On any failure (network issues, invalid JSON, etc.) the engine falls back to
the ``help.show_options`` intent so the rest of the system can respond
gracefully.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

from .io_types import NLUResult


class QwenNLU:
    """Simple wrapper around a Qwen chat completion endpoint."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_url: str | None = None,
    ) -> None:
        # Default to OpenAI-compatible settings so the class works with both
        # official and locally hosted endpoints.  These values can be
        # overridden using environment variables when necessary.
        self.model = model or os.getenv("QWEN_MODEL", "qwen")
        self.api_url = api_url or os.getenv(
            "QWEN_API_URL", "https://api.openai.com/v1/chat/completions"
        )
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    # ------------------------------------------------------------------
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Send ``messages`` to the configured endpoint and return text."""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "messages": messages,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # The class is designed around the OpenAI style chat completion API.
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    def predict(self, utterance: str) -> NLUResult:
        """Return ``NLUResult`` predicted by Qwen for ``utterance``."""

        system_prompt = (
            "You are an intent classifier for the Cap Predictor chatbot. "
            'Your job is to output JSON only, with "intent" and "slots". '
            "Choose the closest intent from this fixed list: "
            "[pipeline.run_daily, pipeline.run_now, data.ingest, "
            "model.train_eval, plots.make_report, explain.decision, "
            "help.show_options]."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utterance},
        ]

        try:
            raw = self._chat(messages)
            data = json.loads(raw.strip())
            intent = data.get("intent")
            slots = data.get("slots") or {}
            # Normalise ticker like entities to upper case.
            if isinstance(slots, dict):
                tickers = slots.get("tickers")
                if isinstance(tickers, list):
                    slots["tickers"] = [str(t).upper() for t in tickers]
                ticker = slots.get("ticker")
                if isinstance(ticker, str):
                    slots["ticker"] = ticker.upper()
        except Exception:
            # Fall back to help intent on any failure so the caller can handle
            # it uniformly.
            return NLUResult(intent="help.show_options", scores=None, slots={}, missing_slots=[])

        return NLUResult(intent=intent, scores=None, slots=slots, missing_slots=[])

