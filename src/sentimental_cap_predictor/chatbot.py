"""Simple command-line chatbot powered by OpenAI models."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import typer

try:  # pragma: no cover - external dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[misc, assignment]


app = typer.Typer(
    help="Interactive chatbot using OpenAI's chat completions API",
)


def _client() -> OpenAI:
    """Return an OpenAI client configured from environment variables."""

    if OpenAI is None:  # pragma: no cover - import guard
        raise typer.BadParameter("openai package is not installed")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable is missing"
        raise typer.BadParameter(msg)
    return OpenAI(api_key=api_key)


def _ask(
    client: OpenAI,
    model: str,
    history: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    """Send the current history to ``model`` and append the reply.

    Returns the model's reply and updated history. Network failures propagate
    to the caller so they can be surfaced to the user.
    """

    resp = client.chat.completions.create(model=model, messages=history)
    reply = resp.choices[0].message.content or ""
    history.append({"role": "assistant", "content": reply})
    return reply, history


def _summarize_decision(main_reply: str, exp_reply: str) -> str:
    """Explain how the final response was selected.

    The function compares outputs from the main and experimental models and
    returns a human-readable explanation describing any differences and which
    response was chosen.
    """

    if main_reply.strip() == exp_reply.strip():
        return f"Both models agree: {main_reply}"
    return (
        "Main model replied: {main}.\nExperimental model replied: {exp}.\n"
        "Decision: opting for the main model's answer because it is the "
        "production model while the experimental model is still under "
        "evaluation."
    ).format(main=main_reply, exp=exp_reply)


@app.command()
def chat(
    main_model: str = "gpt-3.5-turbo",
    experimental_model: str = "gpt-4o-mini",
) -> None:  # pragma: no cover - CLI wrapper
    """Start an interactive chat session consulting two models.

    The chatbot queries both a *main* and an *experimental* model for every
    question. It then reports which answer was chosen and why. Provide a valid
    ``OPENAI_API_KEY`` environment variable before running the command. Type
    ``exit`` or ``quit`` to end the session.
    """

    client = _client()
    main_hist: List[Dict[str, str]] = []
    exp_hist: List[Dict[str, str]] = []
    typer.echo("Chatbot ready. Type 'exit' to quit.")
    while True:
        user = typer.prompt("You")
        if user.strip().lower() in {"exit", "quit"}:
            break
        main_hist.append({"role": "user", "content": user})
        exp_hist.append({"role": "user", "content": user})
        try:
            main_reply, main_hist = _ask(client, main_model, main_hist)
            exp_reply, exp_hist = _ask(client, experimental_model, exp_hist)
        except Exception as exc:  # pragma: no cover - network failure
            typer.echo(f"Error: {exc}")
            break
        summary = _summarize_decision(main_reply, exp_reply)
        typer.echo(f"Bot: {summary}")


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
