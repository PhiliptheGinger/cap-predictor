"""Simple command-line chatbot powered by small Qwen models.

The assistant can run shell commands when prompted. If a model reply starts
with ``CMD:`` followed by a command, the command is executed locally and the
output is appended to the conversation history before asking the model for a
final explanation.
"""

import subprocess

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


SYSTEM_PROMPT = (
    "System: You are a command-line assistant for the sentimental CAP "
    "predictor project. You can execute shell commands for the user. To "
    "run a command, reply with 'CMD: <command>'. After seeing the command "
    "output you should provide a helpful explanation."
)

CLI_USAGE = (
    "System: Available CLI modules include dataset, plots, "
    "modeling.sentiment_analysis and chatbot. They can be invoked with "
    "'python -m sentimental_cap_predictor.<module>'."
)


def _run_shell(command: str) -> str:
    """Execute ``command`` in the system shell and return its output."""

    result = subprocess.run(
        command, shell=True, check=False, capture_output=True, text=True
    )
    return f"{result.stdout}{result.stderr}".strip()


@app.command()
def chat(
    main_model: str = "Qwen/Qwen2-0.5B-Instruct",
    experimental_model: str = "Qwen/Qwen2-0.5B",
) -> None:  # pragma: no cover - CLI wrapper
    """Start an interactive chat session consulting two local models.

    The chatbot queries both a *main* and an *experimental* model for every
    question and reports which answer was chosen and why. Type ``exit`` or
    ``quit`` to end the session.
    """

    main_gen = _get_pipeline(main_model)
    exp_gen = _get_pipeline(experimental_model)
    main_hist: list[str] = [SYSTEM_PROMPT, CLI_USAGE]
    exp_hist: list[str] = [SYSTEM_PROMPT, CLI_USAGE]
    typer.echo("Chatbot ready. Type 'exit' to quit.")
    while True:
        user = typer.prompt("You")
        if user.strip().lower() in {"exit", "quit"}:
            break
        try:
            main_reply, main_hist = _ask(main_gen, main_hist, user)
            if main_reply.startswith("CMD:"):
                cmd = main_reply.removeprefix("CMD:").strip()
                cmd_output = _run_shell(cmd)
                main_hist.append(f"System: Command output:\n{cmd_output}")
                main_reply, main_hist = _ask(
                    main_gen,
                    main_hist,
                    "Command executed.",
                )
            exp_reply, exp_hist = _ask(exp_gen, exp_hist, user)
        except Exception as exc:  # pragma: no cover - model failure
            typer.echo(f"Error: {exc}")
            break
        summary = _summarize_decision(main_reply, exp_reply)
        typer.echo(f"Bot: {summary}")


if __name__ == "__main__":  # pragma: no cover - entry point
    app()
