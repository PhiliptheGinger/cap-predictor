"""Simple command-line chatbot powered by small Qwen models.

The assistant can run shell commands when prompted. If a model reply starts
with ``CMD:`` followed by a command, the command is executed locally and the
output is appended to the conversation history before asking the model for a
final explanation.
"""

import os
import shlex
import subprocess
from pathlib import Path

import typer

app = typer.Typer(
    help="Interactive chatbot using local Hugging Face models",
)


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
    "predictor project. You can execute shell commands for the user. To "
    "run a command, reply with 'CMD: <command>'. After seeing the command "
    "output you should provide a helpful explanation.\n\n"
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


def _run_shell(command: str) -> str:
    """Execute ``command`` in the system shell and return its output.

    Only ``python -m sentimental_cap_predictor.<module>`` commands where
    ``<module>`` is in :data:`ALLOWED_MODULES` are permitted. Any other
    command returns an error message without being executed.
    """

    parts = shlex.split(command)
    if len(parts) >= 3 and parts[0] == "python" and parts[1] == "-m":
        module = parts[2].removeprefix("sentimental_cap_predictor.")
        if (
            parts[2].startswith("sentimental_cap_predictor.")
            and module in ALLOWED_MODULES
        ):
            typer.echo(LOADING_MESSAGES.get(module, "Thinking..."))
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
) -> None:  # pragma: no cover - CLI wrapper
    """Start an interactive chat session using a single local model.

    The chatbot queries an instruct-tuned model for each question. Type ``exit``
    or ``quit`` to end the session.
    """

    generator = _get_pipeline(main_model)
    history: list[str] = [SYSTEM_PROMPT, CLI_USAGE]
    typer.echo("Chatbot ready. Type 'exit' to quit.")
    while True:
        user = typer.prompt("You")
        if user.strip().lower() in {"exit", "quit"}:
            break
        try:
            reply, history = _ask(generator, history, user)
            if reply.startswith("CMD:"):
                cmd = reply.removeprefix("CMD:").strip()
                cmd_output = _run_shell(cmd)
                typer.echo(cmd_output)
                history.append(f"System: Command output:\n{cmd_output}")
                reply, history = _ask(
                    generator,
                    history,
                    "Command executed.",
                )
        except Exception as exc:  # pragma: no cover - model failure
            typer.echo(f"Error: {exc}")
            break
        typer.echo(f"Bot: {reply}")


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
