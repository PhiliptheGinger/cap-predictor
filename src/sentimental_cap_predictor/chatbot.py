# flake8: noqa
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict

from sentimental_cap_predictor.chatbot_nlu import qwen_intent

ASSISTANT_NAME = "Cap Assistant"
ASSISTANT_TAGLINE = (
    "your project-sidekick for data ingest, pipelines, training, and plots."
)

WELCOME_BANNER = f"""
Hi! I'm {ASSISTANT_NAME} — {ASSISTANT_TAGLINE}

I can:
  • Run the pipeline (now or daily)
  • Ingest market data (tickers, period, interval)
  • Train/evaluate models
  • Plot reports
  • Explain why I chose an action

Try:
  - "run the pipeline now"
  - "please run the daily pipeline"
  - "ingest NVDA and AAPL for 5d at 1h"
  - "train and evaluate on AAPL"
  - "plot results for TSLA YTD"
  - "what can you do?"
  - "who are you?"
""".strip()

HELP_TEXT = """
Here's what I can help with right now:

• Pipelines
  - "run the pipeline now"
  - "run the daily pipeline"

• Data ingest
  - "ingest AAPL for 5d at 1h"
  - "pull data for TSLA period 1Y interval 1d"

• Modeling
  - "train and evaluate on NVDA"
  - "run training for AAPL with random seed 7"

• Plots & reports
  - "plot results for AAPL YTD"
  - "generate charts last week for TSLA"

• Explanations
  - "why did you do that?"
  - "explain the last action"

Tip: ask "who are you?" if you want my identity & scope.
""".strip()

ABOUT_TEXT = (
    f"I'm {ASSISTANT_NAME}. I live inside the Cap Predictor project and route "
    "your requests to project actions.\n"
    "Right now I understand plain-English requests for pipelines, data ingest, "
    "training, plotting, and explanations.\n"
    'If you\'re unsure what to say, just ask "what can you do?"'
)


def _summarize_decision(main_reply: str, exp_reply: str) -> str:
    """Explain how the final response was selected.

    Compares outputs from the main and experimental models and returns a
    human-readable explanation describing any differences and which response
    was chosen.
    """
    if main_reply.strip() == exp_reply.strip():
        return f"Both models agree: {main_reply}"
    return (
        "Main model replied: {main}.\n"
        "Experimental model replied: {exp}.\n"
        "Decision: opting for the main model's answer because it is the "
        "production model while the experimental model is still under "
        "evaluation."
    ).format(main=main_reply, exp=exp_reply)


def _run(cmd_list: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.stdout.strip()
    except Exception as e:  # pragma: no cover - subprocess failure
        return f"Command failed: {e}"


def _run_daily_pipeline(ticker: str, period: str, interval: str) -> str:
    """Run the project's daily pipeline for ``ticker`` via subprocess."""
    args = [
        sys.executable,
        "-m",
        "sentimental_cap_predictor.flows.daily_pipeline",
        "run",
        ticker,
        "--period",
        str(period),
        "--interval",
        str(interval),
    ]
    return _run(args)


def dispatch(intent: str, slots: Dict[str, Any]) -> str:
    # Small talk / meta
    if intent == "smalltalk.greeting":
        return f"Hey there! 👋\n\n{HELP_TEXT}"
    if intent == "help.show_options":
        return HELP_TEXT
    if intent == "bot.identity":
        return ABOUT_TEXT
    if intent == "explain.decision":
        return (
            "I choose actions by matching your request to a known intent and "
            "extracting slots (e.g., tickers, period). If I’m unsure, I’ll ask "
            "a clarifying question."
        )

    # Actions (prefer calling your existing CLI modules via subprocess)
    if intent == "pipeline.run_now":
        ticker = slots.get("ticker") or "NVDA"
        period = slots.get("period") or "5y"
        interval = slots.get("interval") or "1d"
        return _run_daily_pipeline(ticker, period, interval)

    if intent == "pipeline.run_daily":
        ticker = slots.get("ticker") or "NVDA"
        period = slots.get("period") or "5y"
        interval = slots.get("interval") or "1d"
        return _run_daily_pipeline(ticker, period, interval)

    if intent == "data.ingest":
        tickers = slots.get("tickers") or slots.get("ticker") or []
        if isinstance(tickers, str):
            tickers = [tickers]
        period = slots.get("period") or "5d"
        interval = slots.get("interval") or "1h"
        # Known-good CLI from your logs:
        # python -m sentimental_cap_predictor.data.ingest NVDA --period 5d --interval 1h
        args = [
            sys.executable,
            "-m",
            "sentimental_cap_predictor.data.ingest",
            *tickers,
            "--period",
            str(period),
            "--interval",
            str(interval),
        ]
        out = _run(args)
        return (
            out or f"Started ingest for {tickers} period={period} interval={interval}."
        )

    if intent == "model.train_eval":
        ticker = slots.get("ticker")
        if not ticker:
            return "Which ticker should I train on? e.g., 'train and evaluate on NVDA'"
        # python -m sentimental_cap_predictor.modeling.train_eval AAPL
        args = [
            sys.executable,
            "-m",
            "sentimental_cap_predictor.modeling.train_eval",
            ticker,
        ]
        out = _run(args)
        return out or f"Training & evaluating on {ticker}."

    if intent == "plots.make_report":
        ticker = slots.get("ticker")
        if not ticker:
            return "Which ticker should I plot? e.g., 'plot results for TSLA YTD'"
        # You previously ran: python -m sentimental_cap_predictor.plots AAPL
        args = [
            sys.executable,
            "-m",
            "sentimental_cap_predictor.plots",
            ticker,
        ]
        out = _run(args)
        return out or f"Generating report for {ticker}."

    # Friendly default if we somehow miss
    return "I didn’t catch a supported request.\n\n" + HELP_TEXT


def _predict_intent(text: str) -> Dict[str, Any]:
    """Predict intent using the Qwen model with a regex fallback."""

    out = None
    try:
        out = qwen_intent.predict(text)
    except Exception:
        # Propagate to fallback below
        out = None
    if not out or not out.get("intent"):
        out = qwen_intent.predict_fallback(text)
    return out


def repl(debug: bool | None = None) -> None:
    """Run an interactive REPL for the Cap Assistant.

    If ``debug`` is ``None`` its value is derived from the ``CHATBOT_DEBUG``
    environment variable. When enabled, the assistant prints the predicted
    intent and slots before responding.
    """

    if debug is None:
        debug_env = os.getenv("CHATBOT_DEBUG", "")
        debug = debug_env.lower() in {"1", "true", "yes", "on"}

    print(WELCOME_BANNER)
    while True:
        try:
            user = input("prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue
        nlu = _predict_intent(user)
        intent = nlu.get("intent") or "help.show_options"
        slots = nlu.get("slots", {})
        if debug:
            # Show what the NLU decided so we can tune prompts
            print(f"[debug] intent={intent} slots={slots}")
        reply = dispatch(intent, slots)
        print(reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cap Assistant chatbot")
    parser.add_argument(
        "--debug", action="store_true", help="Show intent and slot information"
    )
    args = parser.parse_args()
    repl(debug=args.debug)
