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
            "You are an intent classifier and slot extractor for the Cap Predictor CLI.\n"
            "Return ONLY JSON between <json>...</json> tags. No prose.\n"
            "Choose the intent from this FIXED list:\n"
            "[pipeline.run_daily, pipeline.run_now, data.ingest, model.train_eval, plots.make_report, explain.decision, help.show_options]\n\n"
            "Rules:\n"
            "- If the text is clearly outside these intents, use help.show_options.\n"
            "- Extract slots when relevant:\n"
            "  tickers: array of uppercase symbols like AAPL, NVDA (regex [A-Z\\.]{1,5})\n"
            "  period: one of 1D,5D,1M,6M,1Y,5Y,max, or phrases like \"last week\",\"ytd\"\n"
            "  interval: one of 1m,1h,1d or patterns like \\d+m/\\d+h/\\d+d\n"
            "  range: free phrases like \"YTD\",\"last week\"\n"
            "  split: strings like \"80/20\"\n"
            "  seed: integer\n"
            "- Normalize tickers to uppercase. Omit unknown slots.\n"
            "- If unsure between two intents, pick the best AND include \"alt_intent\" with the runner-up.\n"
            "- Absolutely no text outside the <json> block."
        )

        user_prompt = (
            f'Utterance: "{utterance}"\n\n'
            "Few-shot hints (examples):\n"
            "- \"please run the daily pipeline\" -> pipeline.run_daily\n"
            "- \"run the pipeline now\" -> pipeline.run_now\n"
            "- \"ingest NVDA and AAPL for 5d at 1h\" -> data.ingest with slots\n"
            "- \"train and evaluate on NVDA\" -> model.train_eval\n"
            "- \"plot results for AAPL YTD\" -> plots.make_report\n"
            "- \"why did you do that?\" -> explain.decision\n"
            "- \"help me out\" -> help.show_options\n\n"
            "Return exactly:\n<json>\n"
            "{\"intent\": \"...\", \"slots\": {...}, \"alt_intent\": \"...\" }\n"
            "</json>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = self._chat(messages)
            import re

            m = re.search(r"<json>\s*(\{.*?\})\s*</json>", raw, re.S)
            if not m:
                raise ValueError("no json block")
            data = json.loads(m.group(1))
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

