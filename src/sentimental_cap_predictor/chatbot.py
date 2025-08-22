"""Simple command-line chatbot powered by small Qwen models.

The assistant can run shell commands when prompted. If a model reply starts
with ``CMD:`` followed by a command, the command is executed locally and the
output is appended to the conversation history before asking the model for a
final explanation.
"""

import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

import typer

app = typer.Typer(
    help="Interactive chatbot using local Hugging Face models",
)


THEMES = {
    "default": {
        "user": {"fg": typer.colors.CYAN},
        "bot": {"fg": typer.colors.GREEN},
        "system": {"fg": typer.colors.YELLOW},
        "command": {"fg": typer.colors.MAGENTA},
        "error": {"fg": typer.colors.RED},
    },
    "high-contrast": {
        "user": {"fg": typer.colors.BRIGHT_CYAN, "bold": True},
        "bot": {"fg": typer.colors.BRIGHT_GREEN, "bold": True},
        "system": {"fg": typer.colors.BRIGHT_YELLOW, "bold": True},
        "command": {"fg": typer.colors.BRIGHT_MAGENTA, "bold": True},
        "error": {"fg": typer.colors.BRIGHT_RED, "bold": True},
    },
}


def _get_pipeline(model_id: str):
    """Return a text-generation pipeline for ``model_id``."""

    import os

    # Disable optional backends that can trigger heavy imports or
    # incompatibilities on systems where TensorFlow or Flax are installed
    # but not fully configured. ``transformers`` checks the ``USE_TF`` and
    # ``USE_FLAX`` environment variables when deciding whether to import
    # those frameworks. Setting them to ``0`` prevents expensive imports that
    # can lead to protobuf runtime errors on machines that happen to have
    # TensorFlow installed.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    from transformers import pipeline

    return pipeline("text-generation", model=model_id, tokenizer=model_id)


def _ask(generator, history: list[str], user: str) -> tuple[str, list[str]]:
    """Send the updated history to ``generator`` and append its reply."""

    prompt = "\n".join(history + [f"User: {user}", "Assistant:"])
    gen_output = generator(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
    )
    result = gen_output[0]["generated_text"]
    reply = result.split("Assistant:")[-1].strip()
    history.extend([f"User: {user}", f"Assistant: {reply}"])
    return reply, history


MISSION_STATEMENT = (
    "A Model for the Community\n\n"
    "We build this model not only to predict markets, but to reclaim our most "
    "valuable resource: time.\n\n"
    "In today’s world, information is locked away, and efficiency is "
    "hoarded by those with wealth and power. Ordinary people are left to "
    "work longer hours, carrying the burden of complexity without sharing "
    "in the benefits.\n\n"
    "This project is different.\n"
    "It is designed to free people from repetitive, isolating labor — not to "
    "extract profit, but to create space. Space for families, for neighbors, "
    "for art, for rest, for learning, for organizing.\n\n"
    "The insights generated here will be shared through a data cooperative: a "
    "commons where knowledge and tools belong to the community that "
    "creates them. We believe that when data and technology are governed "
    "democratically, they become engines for solidarity rather than "
    "exploitation.\n\n"
    "This model is not an end in itself. It is a step toward a society where "
    "tools serve people, and where freed time can be used to build "
    "something better together.\n\n"
    "We commit to building technology that strengthens community, distributes "
    "power, and helps us imagine new ways of living well."
)


SYSTEM_PROMPT = (
    "System: You are a command-line assistant for the sentimental CAP "
    "predictor project. You can execute shell commands for the user when "
    "explicitly asked. Never claim to have run a command or produced "
    "results unless the command was actually executed. When you show "
    "commands or outputs without running them, make it clear they are "
    "examples for the user to run. To run a command, reply with 'CMD: "
    "<command>'. After seeing the command output you should provide a "
    "helpful explanation.\n\n"
    f"{MISSION_STATEMENT}"
)

CLI_USAGE = (
    "System: Available CLI modules include dataset, data.ingest, "
    "backtest.engine, modeling.sentiment_analysis, modeling.train_eval, plots "
    "and chatbot. They can be invoked with 'python -m "
    "sentimental_cap_predictor.<module>'. Only these commands will be "
    "executed."
)


ALLOWED_MODULES = {
    "dataset",
    "data.ingest",
    "backtest.engine",
    "modeling.sentiment_analysis",
    "modeling.train_eval",
    "plots",
    "chatbot",
}


LOADING_MESSAGES = {
    "dataset": "Analyzing dataset...",
    "data.ingest": "Ingesting data...",
    "backtest.engine": "Running back-testing engine...",
    "modeling.sentiment_analysis": "Analyzing sentiment...",
    "modeling.train_eval": "Training and evaluating model...",
    "plots": "Generating plot...",
    "chatbot": "Thinking...",
}


def _run_shell(
    command: str,
    style: Callable[[str, str], str] | None = None,
) -> str:
    """Execute ``command`` in the system shell and return its output.

    Only ``python -m sentimental_cap_predictor.<module>`` commands where
    ``<module>`` is in :data:`ALLOWED_MODULES` are permitted. Any other
    command returns an error message without being executed.
    """

    if style is None:

        def style_fn(text: str, role: str = "system") -> str:
            return text

        style = style_fn

    parts = shlex.split(command)
    if len(parts) >= 3 and parts[0] == "python" and parts[1] == "-m":
        module = parts[2].removeprefix("sentimental_cap_predictor.")
        if (
            parts[2].startswith("sentimental_cap_predictor.")
            and module in ALLOWED_MODULES
        ):
            message = LOADING_MESSAGES.get(module, "Thinking...")
            typer.echo(style(message, "system"))
            src_dir = Path(__file__).resolve().parents[1]
            project_root = src_dir.parent
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH', '')}"
            result = subprocess.run(
                parts,
                check=False,
                capture_output=True,
                text=True,
                cwd=project_root,
                env=env,
            )
            return f"{result.stdout}{result.stderr}".strip()
    return "Command not allowed."


@app.command()
def chat(
    main_model: str = "Qwen/Qwen2-0.5B-Instruct",
    theme: str = typer.Option(
        "default", help="Color theme for output (default or high-contrast)."
    ),
    no_color: bool = typer.Option(False, help="Disable colored output."),
) -> None:  # pragma: no cover - CLI wrapper
    """Start an interactive chat session using a single local model.

    The chatbot queries an instruct-tuned model for each question.
    Type ``exit`` or ``quit`` to end the session.
    """

    theme_styles = THEMES.get(theme.lower(), THEMES["default"])

    def style(text: str, role: str) -> str:
        if no_color:
            return text
        return typer.style(text, **theme_styles[role])

    generator = _get_pipeline(main_model)
    history: list[str] = [SYSTEM_PROMPT, CLI_USAGE]
    typer.echo(style("Chatbot ready. Type 'exit' to quit.", "system"))
    typer.echo(
        style(
            "Commands are only executed when the assistant "
            "replies with a line starting with 'CMD:'. "
            "Other suggestions are examples and not run automatically.",
            "system",
        )
    )
    while True:
        user = typer.prompt(style("You", "user"))
        if user.strip().lower() in {"exit", "quit"}:
            break
        if user.startswith("CMD:"):
            cmd = user.removeprefix("CMD:").strip()
            cmd_output = _run_shell(cmd, style)
            typer.echo(style(cmd_output, "command"))
            history.extend([
                f"User: CMD: {cmd}",
                f"System: Command output:\n{cmd_output}",
            ])
            continue
        try:
            reply, history = _ask(generator, history, user)
            if reply.startswith("CMD:"):
                cmd = reply.removeprefix("CMD:").strip()
                cmd_output = _run_shell(cmd, style)
                typer.echo(style(cmd_output, "command"))
                history.append(f"System: Command output:\n{cmd_output}")
                reply, history = _ask(
                    generator,
                    history,
                    "Command executed.",
                )
        except Exception as exc:  # pragma: no cover - model failure
            typer.echo(style(f"Error: {exc}", "error"))
            break
        typer.echo(style(f"Bot: {reply}", "bot"))


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
