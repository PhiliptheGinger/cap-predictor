from sentimental_cap_predictor.chatbot import (
    SYSTEM_PROMPT,
    _run_shell,
    _summarize_decision,
)


def test_summarize_decision_agreement():
    result = _summarize_decision("yes", "yes")
    assert "both models agree" in result.lower()


def test_summarize_decision_disagreement():
    result = _summarize_decision("yes", "no")
    assert "main model" in result.lower() and "experimental" in result.lower()


def test_run_shell_executes_command():
    output = _run_shell("python -m sentimental_cap_predictor.chatbot --help")
    assert "usage" in output.lower()


def test_run_shell_rejects_unlisted_command():
    output = _run_shell("echo hello")
    assert "not allowed" in output.lower()


def test_system_prompt_mentions_cli():
    assert "command-line assistant" in SYSTEM_PROMPT.lower()
