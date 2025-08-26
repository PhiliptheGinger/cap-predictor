"""Qwen-based intent classifier returning JSON dicts."""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from openai import OpenAI

    client = OpenAI()
except Exception:  # pragma: no cover - tests monkeypatch client
    class _Dummy:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._missing))

        def _missing(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise RuntimeError("Qwen client not configured")

    client = _Dummy()

SYSTEM = (
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

FEWSHOT = (
    "Few-shot hints (examples):\n"
    "- \"please run the daily pipeline\" -> pipeline.run_daily\n"
    "- \"run the pipeline now\" -> pipeline.run_now\n"
    "- \"ingest NVDA and AAPL for 5d at 1h\" -> data.ingest with slots\n"
    "- \"train and evaluate on NVDA\" -> model.train_eval\n"
    "- \"plot results for AAPL YTD\" -> plots.make_report\n"
    "- \"why did you do that?\" -> explain.decision\n"
    "- \"help me out\" -> help.show_options"
)


def predict(utterance: str) -> Dict[str, Any]:
    """Return intent/slots mapping for ``utterance``."""

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f'Utterance: "{utterance}"\n\n'
                f"{FEWSHOT}\n\n"
                "Return exactly:\n<json>\n"
                '{"intent": "...", "slots": {...}, "alt_intent": "..." }\n'
                "</json>"
            ),
        },
    ]

    try:
        model = os.getenv("QWEN_MODEL", "qwen")
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
        )
        raw = resp.choices[0].message.content
        match = re.search(r"<json>\s*(\{.*?\})\s*</json>", raw, re.S)
        if not match:
            raise ValueError("missing json block")
        return json.loads(match.group(1))
    except Exception:
        return {"intent": "help.show_options", "slots": {}}
