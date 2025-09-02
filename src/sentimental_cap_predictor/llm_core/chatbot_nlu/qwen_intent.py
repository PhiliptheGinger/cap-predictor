from __future__ import annotations

# flake8: noqa
import json
import os
import re as _re
from typing import Any, Dict

from ..config_llm import get_llm_config

SYSTEM = """You are an intent classifier and slot extractor for the Cap Predictor CLI.
Return ONLY JSON between <json>...</json>. Choose the intent from this FIXED list:
[pipeline.run_daily, pipeline.run_now, data.ingest, model.train_eval, plots.make_report,
 explain.decision, help.show_options, bot.identity, smalltalk.greeting, info.lookup,
 info.arxiv, info.pubmed, info.openalex, info.fred, info.github]
Rules:
- If input is outside these intents, use help.show_options.
- Extract slots when relevant: tickers[], period, interval, range, split, seed.
- Normalize tickers to uppercase. Omit unknown slots.
- If unsure, also include "alt_intent" with the runner-up.
"""

FEWSHOT = """Few-shot:
"please run the daily pipeline" -> pipeline.run_daily
"run the pipeline now" -> pipeline.run_now
"ingest NVDA and AAPL for 5d at 1h" -> data.ingest
"train and evaluate on NVDA" -> model.train_eval
"plot results for AAPL YTD" -> plots.make_report
"why did you do that?" -> explain.decision
"what can you do?" -> help.show_options
"who are you?" -> bot.identity
"hey! how's it going?" -> smalltalk.greeting
"news about NVDA" -> info.lookup
"arxiv machine learning" -> info.arxiv
"pubmed cancer research" -> info.pubmed
"openalex reinforcement learning" -> info.openalex
"fred GDP" -> info.fred
"github repo openai/gpt-4" -> info.github
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


_LOCAL_PROVIDER: QwenLocalProvider | None = None


def call_qwen(utterance: str) -> str:
    """Classify ``utterance`` using a local Qwen model."""

    global _LOCAL_PROVIDER
    if _LOCAL_PROVIDER is None:
        from ..llm_providers.qwen_local import (
            QwenLocalProvider,
        )

        cfg = get_llm_config()
        # Use deterministic settings for intent classification
        _LOCAL_PROVIDER = QwenLocalProvider(model_path=cfg.model_path, temperature=0.0)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _build_user_prompt(utterance)},
    ]
    return _LOCAL_PROVIDER.chat(messages, max_new_tokens=200, top_p=1.0)


_JSON_RE = _re.compile(r"<json>\s*(\{.*\})\s*</json>", _re.S)


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
    except Exception:  # pragma: no cover - model inference failure
        return None

    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# --- Minimal keyword fallback (used if Qwen not yet wired) ---

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

_ARXIV = _re.compile(r"\barxiv\b(?:\s+(?:about|on))?\s+(.+)", _re.I)
_PUBMED = _re.compile(r"\bpubmed\b(?:\s+(?:about|on))?\s+(.+)", _re.I)
_OPENALEX = _re.compile(r"\bopenalex\b(?:\s+(?:about|on))?\s+(.+)", _re.I)
_FRED = _re.compile(r"\bfred\b.*?([A-Za-z0-9_]+)", _re.I)
_GITHUB = _re.compile(r"\bgithub\s+(?:repo\s+)?([\w-]+)/([\w.-]+)", _re.I)

# Generic information lookups (e.g., "news about NVDA", "lookup inflation data")
_INFO_LOOKUP = _re.compile(
    r"\b(?:news|info|information|lookup|look up)\b(?:\s+(?:about|on))?\s+(.+)",
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
    m = _ARXIV.search(text)
    if m:
        return {"intent": "info.arxiv", "slots": {"query": m.group(1).strip()}}
    m = _PUBMED.search(text)
    if m:
        return {"intent": "info.pubmed", "slots": {"query": m.group(1).strip()}}
    m = _OPENALEX.search(text)
    if m:
        return {"intent": "info.openalex", "slots": {"query": m.group(1).strip()}}
    m = _FRED.search(text)
    if m:
        return {
            "intent": "info.fred",
            "slots": {"series_id": m.group(1).upper()},
        }
    m = _GITHUB.search(text)
    if m:
        return {
            "intent": "info.github",
            "slots": {"owner": m.group(1), "repo": m.group(2)},
        }
    m = _INFO_LOOKUP.search(text)
    if m:
        return {"intent": "info.lookup", "slots": {"query": m.group(1).strip()}}
    return {"intent": "help.show_options", "slots": {}}
