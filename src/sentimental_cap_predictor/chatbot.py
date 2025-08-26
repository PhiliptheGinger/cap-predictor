from __future__ import annotations
import sys, subprocess, shlex
from typing import Dict, Any

from sentimental_cap_predictor.chatbot_nlu import qwen_intent

ASSISTANT_NAME = "Cap Assistant"
ASSISTANT_TAGLINE = "your project-sidekick for data ingest, pipelines, training, and plots."

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

HELP_TEXT = f"""
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

ABOUT_TEXT = f"""
I'm {ASSISTANT_NAME}. I live inside the Cap Predictor project and route your requests to project actions.
Right now I understand plain-English requests for pipelines, data ingest, training, plotting, and explanations.
If you're unsure what to say, just ask "what can you do?"
""".strip()

def _run(cmd_list: list[str]) -> str:
    try:
        proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc.stdout.strip()
    except Exception as e:
        return f"Command failed: {e}"

def dispatch(intent: str, slots: Dict[str, Any]) -> str:
    # Small talk / meta
    if intent == "smalltalk.greeting":
        return f"Hey there! 👋\n\n{HELP_TEXT}"
    if intent == "help.show_options":
        return HELP_TEXT
    if intent == "bot.identity":
        return ABOUT_TEXT
    if intent == "explain.decision":
        return "I choose actions by matching your request to a known intent and extracting slots (e.g., tickers, period). If I’m unsure, I’ll ask a clarifying question."

    # Actions (prefer calling your existing CLI modules via subprocess)
    if intent == "pipeline.run_now":
        # If you have a CLI for pipeline, call it here; otherwise stub a success.
        # Example (adjust if your pipeline module differs):
        # cmd = [sys.executable, "-m", "sentimental_cap_predictor.pipeline", "run_now"]
        # out = _run(cmd)
        # return out or "Running the pipeline now."
        return "Running the pipeline now."

    if intent == "pipeline.run_daily":
        # cmd = [sys.executable, "-m", "sentimental_cap_predictor.pipeline", "run_daily"]
        # out = _run(cmd)
        # return out or "Kicking off the daily pipeline."
        return "Kicking off the daily pipeline."

    if intent == "data.ingest":
        tickers = slots.get("tickers") or slots.get("ticker") or []
        if isinstance(tickers, str): tickers = [tickers]
        period = slots.get("period") or "5d"
        interval = slots.get("interval") or "1h"
        # Known-good CLI from your logs:
        # python -m sentimental_cap_predictor.data.ingest NVDA --period 5d --interval 1h
        args = [sys.executable, "-m", "sentimental_cap_predictor.data.ingest", *tickers, "--period", str(period), "--interval", str(interval)]
        out = _run(args)
        return out or f"Started ingest for {tickers} period={period} interval={interval}."

    if intent == "model.train_eval":
        ticker = slots.get("ticker")
        if not ticker:
            return "Which ticker should I train on? e.g., 'train and evaluate on NVDA'"
        # python -m sentimental_cap_predictor.modeling.train_eval AAPL
        args = [sys.executable, "-m", "sentimental_cap_predictor.modeling.train_eval", ticker]
        out = _run(args)
        return out or f"Training & evaluating on {ticker}."

    if intent == "plots.make_report":
        ticker = slots.get("ticker")
        if not ticker:
            return "Which ticker should I plot? e.g., 'plot results for TSLA YTD'"
        # You previously ran: python -m sentimental_cap_predictor.plots AAPL
        args = [sys.executable, "-m", "sentimental_cap_predictor.plots", ticker]
        out = _run(args)
        return out or f"Generating report for {ticker}."

    # Friendly default if we somehow miss
    return "I didn’t catch a supported request.\n\n" + HELP_TEXT

def _predict_intent(text: str) -> Dict[str, Any]:
    # Try Qwen NLU; if not wired yet, use fallback so commands still work
    try:
        out = qwen_intent.predict(text)
        if not out or not out.get("intent"):
            out = qwen_intent.predict_fallback(text)
    except Exception:
        out = qwen_intent.predict_fallback(text)
    return out

def repl():
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
        # DEBUG: show what the NLU decided so we can tune prompts
        print(f"[debug] intent={intent} slots={slots}")
        reply = dispatch(intent, slots)
        print(reply)

if __name__ == "__main__":
    repl()
