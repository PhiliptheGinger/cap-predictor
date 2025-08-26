from __future__ import annotations

# flake8: noqa
import json
import os
import re
from typing import Any, Dict

SYSTEM = """You are an intent classifier and slot extractor for the Cap Predictor CLI.
Return ONLY JSON between <json>...</json>. Choose the intent from this FIXED list:
[pipeline.run_daily, pipeline.run_now, data.ingest, model.train_eval, plots.make_report, explain.decision, help.show_options, bot.identity, smalltalk.greeting]
Rules:
- If input is outside these intents, use help.show_options.
- Extract slots when relevant: tickers[], period, interval, range, split, seed.
- Normalize tickers to uppercase. Omit unknown slots.
- If unsure, also include "alt_intent" with the runner-up.
"""

FEWSHOT = """Few-shot:
- "please run the daily pipeline" -> pipeline.run_daily
- "run the pipeline now" -> pipeline.run_now
- "ingest NVDA and AAPL for 5d at 1h" -> data.ingest
- "train and evaluate on NVDA" -> model.train_eval
- "plot results for AAPL YTD" -> plots.make_report
- "why did you do that?" -> explain.decision
- "what can you do?" -> help.show_options
- "who are you?" -> bot.identity
- "hey! how's it going?" -> smalltalk.greeting
"""


def _build_user_prompt(utterance: str) -> str:
    return (
        f'Utterance: "{utterance}"\n'
        f"{FEWSHOT}\n"
        "Return exactly:\n"
        "<json>\n"
        '{"intent":"...", "slots":{}, "alt_intent":"..."}\n'
        "</json>"
    )


def call_qwen(utterance: str) -> str:
    """Call the Qwen API to classify ``utterance``.

    The Qwen SDK is expected to look for an API key in the ``DASHSCOPE_API_KEY``
    environment variable.  The ``dashscope`` package provides a ``Generation``
    endpoint compatible with OpenAI-style chat messages.  We pass the
    :data:`SYSTEM` prompt as ``system`` and the constructed user prompt from
    :func:`_build_user_prompt` as ``user``.  The model is queried with
    deterministic settings (``temperature=0.0`` and ``top_p=1.0``) and the raw
    text output is returned.
    """

    from dashscope import Generation

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_user_prompt(utterance)},
    ]

    response = Generation.call(
        model=os.getenv("QWEN_INTENT_MODEL", "qwen-turbo"),
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
        result_format="text",
    )

    if response.status_code == 200:
        output = response.output or {}
        if isinstance(output, dict):
            return output.get("text", str(output))
        return str(output)
    msg = getattr(response, "message", "Qwen API error")
    raise RuntimeError(f"Qwen request failed: {msg}")


_JSON_RE = re.compile(r"<json>\s*(\{.*\})\s*</json>", re.S)


def predict(utterance: str) -> Dict[str, Any] | None:
    """Call Qwen to classify an utterance.

    Returns the parsed intent dictionary on success. If the model response
    doesn't contain a JSON block, ``None`` is returned. Any exception from the
    underlying model call is allowed to propagate so the caller can decide how
    to handle it (e.g., by invoking a keyword fallback).
    """

    text = call_qwen(utterance)
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    return json.loads(m.group(1))


# --- Minimal keyword fallback (used if Qwen not yet wired) ---
import re as _re  # noqa: E402

_PIPELINE_NOW = _re.compile(
    r"\b(run|start|kick off|execute).*(pipeline|flow).*(now|right away|immediately)?",
    _re.I,
)
_PIPELINE_DAILY = _re.compile(r"\b(daily)\b|\bevery\s*day\b", _re.I)
_HELP = _re.compile(r"\b(help|what can you do|how do i|commands?)\b", _re.I)
_WHO = _re.compile(r"\b(who are you|what are you|what is this)\b", _re.I)
_HELLO = _re.compile(r"\b(hi|hello|hey|how'?s it going|what'?s up)\b", _re.I)
_INGEST = _re.compile(r"\b(ingest|pull|fetch|download).*(\b[A-Z\.]{1,5}\b)", _re.I)


def predict_fallback(utterance: str) -> Dict[str, Any]:
    text = utterance.strip()
    if _HELLO.search(text):
        return {"intent": "smalltalk.greeting", "slots": {}}
    if _WHO.search(text):
        return {"intent": "bot.identity", "slots": {}}
    if _HELP.search(text):
        return {"intent": "help.show_options", "slots": {}}
    if _PIPELINE_DAILY.search(text):
        return {"intent": "pipeline.run_daily", "slots": {}}
    if _PIPELINE_NOW.search(text):
        return {"intent": "pipeline.run_now", "slots": {}}
    if _INGEST.search(text):
        return {"intent": "data.ingest", "slots": {}}
    return {"intent": "help.show_options", "slots": {}}
