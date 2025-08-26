from unittest.mock import patch

from sentimental_cap_predictor import chatbot
from sentimental_cap_predictor.chatbot import _summarize_decision


def test_summarize_decision_agreement():
    result = _summarize_decision("yes", "yes")
    assert "both models agree" in result.lower()


def test_summarize_decision_disagreement():
    result = _summarize_decision("yes", "no")
    assert "main model" in result.lower() and "experimental" in result.lower()


def _run_repl_once(debug=None):
    with patch("builtins.input", side_effect=["hi", EOFError()]):
        with patch(
            "sentimental_cap_predictor.chatbot._predict_intent",
            return_value={"intent": "smalltalk.greeting", "slots": {}},
        ):
            chatbot.repl(debug=debug)


def test_repl_no_debug_output(monkeypatch, capsys):
    monkeypatch.delenv("CHATBOT_DEBUG", raising=False)
    _run_repl_once()
    captured = capsys.readouterr()
    assert "[debug]" not in captured.out


def test_repl_debug_output_env(monkeypatch, capsys):
    monkeypatch.setenv("CHATBOT_DEBUG", "1")
    _run_repl_once()
    captured = capsys.readouterr()
    assert "[debug] intent=smalltalk.greeting" in captured.out
