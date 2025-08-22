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
        line = f"- {name}"
        if summary:
            line += f": {summary}"
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
        prompt = prompt_fn("prompt")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        if prompt.strip().lower() == "help":
            _print_help(nl_parser, echo_fn)
            continue
        try:
            task = nl_parser.parse(prompt)
        except Exception as exc:  # pragma: no cover - parser failure
            _handle_error(exc, debug, echo_fn)
            continue
        if _needs_confirmation(task):
            if not confirm_fn("Execute?", default=False):
                echo_fn("Cancelled")
                continue
        try:
            result = dispatcher.dispatch(task)
        except Exception as exc:  # pragma: no cover - dispatcher failure
            _handle_error(exc, debug, echo_fn)
            continue
        _print_result(result, echo_fn)


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
    """Launch interactive chatbot using project parser and dispatcher.

    The function attempts to import ``nl_parser`` and ``dispatcher`` from the
    package.  If they are missing an informative message is shown.
    """

    try:
        from . import dispatcher as default_dispatcher  # type: ignore
        from . import nl_parser as default_parser  # type: ignore
    except Exception as exc:  # pragma: no cover - import failure
        typer.echo(f"Unable to import parser/dispatcher: {exc}")
        return

    chat_loop(default_parser, default_dispatcher, debug=debug)


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
