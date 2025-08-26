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
# Typer entry point
# ---------------------------------------------------------------------------


@app.command()
def chat(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show tracebacks on errors",
    ),
) -> None:  # pragma: no cover - CLI wrapper
    """Launch interactive chatbot using the intent/slot engine."""

    try:  # pragma: no cover - import failure
        from . import chatbot_nlu as bot
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Unable to import NLU components: {exc}")
        return

    ctx: dict = {}
    while True:
        prompt = typer.prompt("prompt")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        try:
            nlu = bot.parse(prompt, ctx)
            res = bot.resolve(nlu, ctx)
            if res.action_needed == "ASK_CLARIFY" and res.prompt:
                typer.echo(res.prompt)
                choice = typer.prompt("choice")
                nlu = bot.parse(choice, ctx)
                res = bot.resolve(nlu, ctx)
            if res.action_needed == "ASK_SLOT" and res.prompt:
                for slot in nlu.missing_slots:
                    val = typer.prompt(slot)
                    res.slots[slot] = val
                res.action_needed = "DISPATCH"
            if res.action_needed == "FALLBACK":
                typer.echo(res.prompt or "Sorry, I can't help with that.")
                continue
            decision = bot.dispatch(res, ctx)
            argument = bot.explain(decision, nlu, ctx)
            summary = decision.result.get("summary") if isinstance(decision.result, dict) else None
            if summary:
                typer.echo(f"SUCCESS: {summary}")
            typer.echo(argument.text)
        except Exception as exc:  # pragma: no cover
            typer.echo(f"Error: {exc}")
            if debug:
                traceback.print_exc()


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
