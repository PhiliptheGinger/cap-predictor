# flake8: noqa
from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict

from sentimental_cap_predictor.chatbot_nlu import qwen_intent
from sentimental_cap_predictor.data.news import fetch_news
from sentimental_cap_predictor.connectors import (
    arxiv_connector,
    fred_connector,
    github_connector,
    openalex_connector,
    pubmed_connector,
)

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
  • Look up recent news or papers
  • Query arXiv, PubMed, OpenAlex, FRED and GitHub
  • Explain why I chose an action

Try:
  - "run the pipeline now"
  - "please run the daily pipeline"
  - "ingest NVDA and AAPL for 5d at 1h"
  - "train and evaluate on AAPL"
  - "plot results for TSLA YTD"
  - "news about NVDA"
  - "arxiv machine learning"
  - "pubmed cancer research"
  - "openalex reinforcement learning"
  - "fred GDP"
  - "github repo openai/gpt-4"
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

• Info lookup
  - "news about NVDA"
  - "arxiv machine learning"
  - "pubmed cancer research"
  - "openalex reinforcement learning"
  - "fred GDP"
  - "github repo openai/gpt-4"

• Explanations
  - "why did you do that?"
  - "explain the last action"

Tip: ask "who are you?" if you want my identity & scope.
""".strip()

ABOUT_TEXT = (
    f"I'm {ASSISTANT_NAME}. I live inside the Cap Predictor project and route "
    "your requests to project actions.\n"
    "Right now I understand plain-English requests for pipelines, data ingest, "
    "training, plotting, explanations, and information lookup.\n"
    'If you\'re unsure what to say, just ask "what can you do?"'
)


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


def _run_pipeline_from_slots(slots: Dict[str, Any]) -> str:
    """Build and run the daily pipeline command based on NLU slots."""
    ticker = slots.get("ticker") or "NVDA"
    period = slots.get("period") or "5y"
    interval = slots.get("interval") or "1d"
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


def _lookup_arxiv(query: str) -> str:
    try:
        papers = arxiv_connector.fetch(query=query, max_results=5)
    except Exception as exc:  # pragma: no cover - network failure
        return f"Error fetching arXiv results for '{query}': {exc}"
    if not papers:
        return f"No arXiv results for '{query}'."
    lines = [f"{p['title']} ({p.get('updated', '')[:10]})" for p in papers[:5]]
    return "Top arXiv hits:\n" + "\n".join(f"- {l}" for l in lines)


def _lookup_pubmed(query: str) -> str:
    try:
        articles = pubmed_connector.fetch(query=query, max_results=5)
    except Exception as exc:  # pragma: no cover - network failure
        return f"Error fetching PubMed results for '{query}': {exc}"
    if not articles:
        return f"No PubMed results for '{query}'."
    lines = [a.get('title', '') for a in articles[:5]]
    return "Top PubMed hits:\n" + "\n".join(f"- {l}" for l in lines)


def _lookup_openalex(search: str) -> str:
    try:
        works = openalex_connector.fetch(search=search, per_page=5)
    except Exception as exc:  # pragma: no cover - network failure
        return f"Error fetching OpenAlex results for '{search}': {exc}"
    if not works:
        return f"No OpenAlex results for '{search}'."
    lines = [w.get('title', '') for w in works[:5]]
    return "Top OpenAlex hits:\n" + "\n".join(f"- {l}" for l in lines)


def _lookup_fred(series_id: str) -> str:
    try:
        observations = fred_connector.fetch_series(series_id, limit=5)
    except TypeError:
        # older fetch_series signature
        try:
            observations = fred_connector.fetch_series(series_id)
        except Exception as exc:  # pragma: no cover - network failure
            return f"Error fetching FRED series '{series_id}': {exc}"
    except Exception as exc:  # pragma: no cover - network failure
        return f"Error fetching FRED series '{series_id}': {exc}"
    if not observations:
        return f"No data found for FRED series '{series_id}'."
    lines = [f"{o.get('date')}: {o.get('value')}" for o in observations[:5]]
    return "FRED observations:\n" + "\n".join(f"- {l}" for l in lines)


def _lookup_github(owner: str, repo: str) -> str:
    try:
        data = github_connector.fetch_repo(owner, repo)
    except Exception as exc:  # pragma: no cover - network failure
        return f"Error fetching repo {owner}/{repo}: {exc}"
    name = data.get('full_name', f"{owner}/{repo}")
    desc = data.get('description') or ""
    return f"{name}: {desc}"


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
        return _run_pipeline_from_slots(slots)

    if intent == "pipeline.run_daily":
        return _run_pipeline_from_slots(slots)

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

    if intent == "info.lookup":
        query = slots.get("query")
        if not query:
            return "What topic should I look up? e.g., 'news about NVDA'"
        try:
            df = fetch_news(query)
        except Exception as exc:  # pragma: no cover - network failure
            return f"Error fetching info for {query}: {exc}"
        if df.empty:
            return f"No news found for '{query}'."
        lines = [
            f"{row.date.date()}: {row.headline} ({row.source})"
            for _, row in df.head(5).iterrows()
        ]
        return "Here are some headlines:\n" + "\n".join(f"- {line}" for line in lines)

    if intent == "info.arxiv":
        query = slots.get("query")
        if not query:
            return "What should I search on arXiv? e.g., 'arxiv machine learning'"
        return _lookup_arxiv(query)

    if intent == "info.pubmed":
        query = slots.get("query")
        if not query:
            return "What topic should I search on PubMed?"
        return _lookup_pubmed(query)

    if intent == "info.openalex":
        query = slots.get("query")
        if not query:
            return "What should I search on OpenAlex?"
        return _lookup_openalex(query)

    if intent == "info.fred":
        series_id = slots.get("series_id")
        if not series_id:
            return "Which FRED series? e.g., 'fred GDP'"
        return _lookup_fred(series_id)

    if intent == "info.github":
        owner = slots.get("owner")
        repo = slots.get("repo")
        if not owner or not repo:
            return "Which repository? e.g., 'github repo openai/gpt-4'"
        return _lookup_github(owner, repo)

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
