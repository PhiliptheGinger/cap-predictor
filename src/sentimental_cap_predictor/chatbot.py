"""Minimal command line chatbot orchestrating project actions.

The chatbot is intentionally lightweight.  It relies on two collaborators:

``nl_parser``
    Object with a ``parse`` method that converts natural language prompts
    into *tasks*.  The parser is also expected to expose a ``registry``
    attribute describing the available commands.  Registry entries may have
    ``summary`` and ``examples`` fields which are displayed in the help text.

``dispatcher``
    Object with a ``dispatch`` method.  The chatbot hands the parsed task to
    the dispatcher which performs the action and returns an object or mapping
    containing ``summary`` along with optional ``metrics`` and ``artifacts``.

The interactive loop therefore looks like::

    prompt -> nl_parser.parse() -> optional confirmation ->
    dispatcher.dispatch()

Any errors are displayed without a traceback to keep the interface friendly.
Use ``--debug`` to show full tracebacks when diagnosing failures.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

import typer

from .chatbot_nlu import qwen_intent
from .flows.daily_pipeline import run as _run_pipeline


# --- Friendly identity & help text ---
ASSISTANT_NAME = "Cap Assistant"
ASSISTANT_TAGLINE = (
    "your project-sidekick for data ingest, pipelines, training, and plots."
)

WELCOME_BANNER = f"""
Hi! I'm {ASSISTANT_NAME} — {ASSISTANT_TAGLINE}

I can:
  • Run the pipeline (now or on schedule)
  • Ingest market data (tickers, period, interval)
  • Train/evaluate models
  • Plot reports
  • Explain why I chose an action

Try one of these:
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


app = typer.Typer(help="Interactive helper for project utilities")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_attr(obj: Any, name: str, default: Any | None = None) -> Any:
    """Return ``name`` from ``obj`` supporting both dicts and attributes."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _needs_confirmation(task: Any) -> bool:
    """Determine whether ``task`` requests user confirmation."""

    for field in ("confirm", "requires_confirmation", "needs_confirmation"):
        val = _get_attr(task, field, False)
        if val:
            return True
    return False


def _print_result(
    result: Any,
    echo_fn: Callable[[str], None] = typer.echo,
) -> None:
    """Pretty print execution ``result``."""

    summary = _get_attr(result, "summary")
    if summary:
        echo_fn(f"SUCCESS: {summary}")
    else:
        message = _get_attr(result, "message")
        if message:
            echo_fn(f"SUCCESS: {message}")

    metrics = _get_attr(result, "metrics", {}) or {}
    if metrics:
        echo_fn("Metrics:")
        for key, value in metrics.items():
            echo_fn(f"  {key}: {value}")

    artifacts = _get_attr(result, "artifacts", []) or []
    if artifacts:
        echo_fn("Artifacts:")
        for art in artifacts:
            echo_fn(f"  {art}")


def _print_failure(
    result: Any,
    echo_fn: Callable[[str], None] = typer.echo,
) -> None:
    """Print the failure message from ``result``."""

    message = _get_attr(result, "message", "")
    if message:
        echo_fn(message)


def _print_help(
    nl_parser: Any,
    echo_fn: Callable[[str], None] = typer.echo,
) -> None:
    """Display registry information from ``nl_parser``."""

    registry = getattr(nl_parser, "registry", {}) or {}
    if not registry:
        echo_fn("No registered actions.")
        return

    echo_fn("Available commands:")
    for name, entry in registry.items():
        summary = _get_attr(entry, "summary", "")
        examples = _get_attr(entry, "examples", []) or []
        line = f"- {summary}" if summary else f"- {name}"
        echo_fn(line)
        for ex in examples:
            echo_fn(f"  e.g. {ex}")


def _handle_error(
    exc: Exception,
    debug: bool,
    echo_fn: Callable[[str], None] = typer.echo,
) -> None:
    """Render an exception according to ``debug`` flag."""

    echo_fn(f"Error: {exc}")
    if debug:
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def chat_loop(
    nl_parser: Any,
    dispatcher: Any,
    *,
    debug: bool = False,
    prompt_fn: Callable[[str], str] = typer.prompt,
    echo_fn: Callable[[str], None] = typer.echo,
    confirm_fn: Callable[[str], bool] = typer.confirm,
) -> None:
    """Run the interactive chatbot loop."""

    while True:
        try:
            prompt = prompt_fn("prompt")
        except StopIteration:  # pragma: no cover - test iter exhausted
            break
        normalized = prompt.strip().lower()
        if normalized in {"exit", "quit"}:
            break
        normalized_no_q = normalized.rstrip("?")
        help_phrases = {"what can you do", "what actions can you take"}
        if (
            normalized in {"help", "?"}
            or any(phrase in normalized_no_q for phrase in help_phrases)
        ):
            _print_help(nl_parser, echo_fn)
            continue
        try:
            tasks = nl_parser.parse(prompt)
        except Exception as exc:  # pragma: no cover - parser failure
            _handle_error(exc, debug, echo_fn)
            continue
        task_list = (
            tasks
            if isinstance(tasks, list)
            else ([tasks] if tasks is not None else [])
        )
        if not task_list or all(not _get_attr(t, "command") for t in task_list):
            echo_fn("Unknown command, type `help` to see options.")
            _print_help(nl_parser, echo_fn)
            continue
        multi = len(task_list) > 1

        for idx, task in enumerate(task_list, 1):
            # show command information and gather any missing parameters
            command_name = _get_attr(task, "command")
            params = _get_attr(task, "params", {}) or {}
            registry = getattr(nl_parser, "registry", {}) or {}
            entry = registry.get(command_name) if command_name else None
            if entry:
                summary = _get_attr(entry, "summary", "")
                schema = _get_attr(entry, "params_schema", {}) or {}
                if summary:
                    echo_fn(f"{command_name}: {summary}")
                required = [
                    name
                    for name, type_str in schema.items()
                    if "None" not in str(type_str)
                ]
                if required:
                    echo_fn("Required params: " + ", ".join(required))
                missing = [n for n in required if n not in params]
                for name in missing:
                    params[name] = prompt_fn(name)
                if isinstance(task, dict):
                    task["params"] = params
                else:
                    setattr(task, "params", params)

            if _needs_confirmation(task):
                if entry:
                    param_line = ", ".join(f"{k}={v}" for k, v in params.items())
                    if summary:
                        echo_fn(
                            f"About to execute {command_name}: {summary}"
                            + (f" with params: {param_line}" if param_line else "")
                        )
                if not confirm_fn("Execute?", default=False):
                    echo_fn("Cancelled")
                    continue
            try:
                result = dispatcher.dispatch(task)
            except Exception as exc:  # pragma: no cover - dispatcher failure
                _handle_error(exc, debug, echo_fn)
                continue
            ok = _get_attr(result, "ok", True)
            if multi:
                echo_fn(f"Step {idx}:")
            if ok:
                _print_result(result, echo_fn)
            else:
                _print_failure(result, echo_fn)


# ---------------------------------------------------------------------------
# Intent dispatcher & Typer entry point
# ---------------------------------------------------------------------------


def dispatch(intent: str, slots: dict) -> str:
    if intent == "pipeline.run_daily":
        _run_pipeline("AAPL")
        return "Kicking off the daily pipeline. I’ll let you know when it completes."
    if intent == "pipeline.run_now":
        _run_pipeline("AAPL")
        return "Running the pipeline now."
    if intent == "data.ingest":
        return (
            f"Starting ingest for {slots.get('tickers')} period={slots.get('period')} interval={slots.get('interval')}."
        )
    if intent == "model.train_eval":
        return f"Training & evaluating on {slots.get('ticker')}."
    if intent == "plots.make_report":
        return f"Generating report for {slots.get('ticker')} range={slots.get('range')}."
    if intent == "explain.decision":
        return "I explain my choices by pointing to the intent I matched, the slots I extracted, and recent context."
    if intent in ("help.show_options", "help", "unknown"):
        return HELP_TEXT
    if intent in ("bot.identity", "who_are_you"):
        return ABOUT_TEXT
    if intent == "smalltalk.greeting":
        return f"Hey there! 👋\n\n{HELP_TEXT}"

    return "I didn’t catch a supported request.\n\n" + HELP_TEXT


def main(*, debug: bool = False) -> None:
    print(WELCOME_BANNER)
    while True:
        prompt = typer.prompt("prompt")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        try:
            data = qwen_intent.predict(prompt)
            intent = data.get("intent", "help.show_options")
            slots = data.get("slots", {}) or {}
            reply = dispatch(intent, slots)
            typer.echo(reply)
        except Exception as exc:  # pragma: no cover
            typer.echo(f"Error: {exc}")
            if debug:
                traceback.print_exc()


@app.command()
def chat(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show tracebacks on errors",
    ),
) -> None:  # pragma: no cover - CLI wrapper
    """Simple chatbot that routes intents to project functions."""

    main(debug=debug)


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
