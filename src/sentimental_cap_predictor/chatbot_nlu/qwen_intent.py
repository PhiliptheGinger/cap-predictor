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
    doesn't contain a JSON block, ``None`` is returned. When the Qwen call
    itself fails, ``None`` is returned so the caller can decide how to handle
    the failure (e.g., by invoking a keyword fallback).  No default intent is
    forced here.
    """

    try:
        text = call_qwen(utterance)
    except Exception:  # pragma: no cover - network/client failure
        return None

    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


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
_TICKER_TOKEN = r"[A-Z\.]{1,5}"
_TICKERS = _re.compile(_TICKER_TOKEN)

# Capture one or more ticker symbols appearing anywhere after the trigger word.
_INGEST = _re.compile(
    rf"\b(ingest|pull|fetch|download)\b.*?({_TICKER_TOKEN}(?:[\s,]+{_TICKER_TOKEN})*)",
    _re.I,
)
_TRAIN = _re.compile(
    rf"\b(train|fit|learn|evaluate)\b.*?({_TICKER_TOKEN}(?:[\s,]+{_TICKER_TOKEN})*)",
    _re.I,
)
_PLOT = _re.compile(
    rf"\b(plot|graph|chart)\b.*?({_TICKER_TOKEN}(?:[\s,]+{_TICKER_TOKEN})*)",
    _re.I,
)


def _slots_from_match(m: _re.Match[str]) -> Dict[str, Any]:
    tickers = [t.upper() for t in _TICKERS.findall(m.group(2))]
    if not tickers:
        return {}
    if len(tickers) == 1:
        return {"ticker": tickers[0]}
    return {"tickers": tickers}


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
    m = _INGEST.search(text)
    if m:
        return {"intent": "data.ingest", "slots": _slots_from_match(m)}
    m = _TRAIN.search(text)
    if m:
        return {"intent": "model.train_eval", "slots": _slots_from_match(m)}
    m = _PLOT.search(text)
    if m:
        return {"intent": "plots.make_report", "slots": _slots_from_match(m)}
    return {"intent": "help.show_options", "slots": {}}
