import types

import pytest

from sentimental_cap_predictor.agent import nl_parser
from sentimental_cap_predictor.chatbot import _print_help, chat_loop


class DummyParser:
    """Very small parser used for testing the chatbot loop."""

    def __init__(self, confirm: bool = False):
        self.confirm = confirm
        self.registry = {
            "foo": {"summary": "do foo", "examples": ["foo bar"]},
        }

    def parse(self, prompt: str) -> dict[str, object]:
        data: dict[str, object] = {"command": prompt}
        if self.confirm:
            data["confirm"] = True
        return data


class DummyDispatcher:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def dispatch(self, task: object) -> dict[str, object]:
        self.dispatched.append(task)
        return {
            "summary": "ok",
            "metrics": {"acc": 1},
            "artifacts": ["a.txt"],
        }


def iter_inputs(*items: str) -> types.FunctionType:
    it = iter(items)

    def _next(_: str) -> str:
        return next(it)

    return _next


def test_print_help_includes_known_commands(capsys):
    intent = nl_parser.parse("help")
    assert intent.command is None
    _print_help(nl_parser)
    out = capsys.readouterr().out
    assert "Download and prepare price data" in out
    assert "Train baseline models and write evaluation CSVs" in out


def _collect_help_output(trigger: str, capsys):
    dispatcher = DummyDispatcher()
    chat_loop(nl_parser, dispatcher, prompt_fn=iter_inputs(trigger, "exit"))
    return capsys.readouterr().out


def test_conversational_help_matches_help(capsys):
    out_help = _collect_help_output("help", capsys)
    out_question = _collect_help_output("what can you do?", capsys)
    assert out_help == out_question
    assert "Download and prepare price data" in out_help


@pytest.mark.parametrize("trigger", ["help", "?"])
def test_help_lists_registry(trigger, capsys):
    parser = DummyParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs(trigger, "exit"))
    out = capsys.readouterr().out
    assert "do foo" in out and "foo bar" in out


@pytest.mark.parametrize(
    "trigger",
    [
        "what can you do?",
        "what actions can you take?",
        "hey, what can you do?",
    ],
)
def test_question_triggers_help(trigger, capsys):
    parser = DummyParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs(trigger, "exit"))
    out = capsys.readouterr().out
    assert "do foo" in out and "foo bar" in out
    assert "Unknown command" not in out


def test_dispatch_and_prints(capsys):
    parser = DummyParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("run", "exit"))
    out = capsys.readouterr().out
    assert "SUCCESS: ok" in out
    assert "acc" in out and "a.txt" in out
    assert dispatcher.dispatched


def test_dispatch_uses_message_when_no_summary(capsys):
    class MessageDispatcher(DummyDispatcher):
        def dispatch(  # type: ignore[override]
            self,
            task: object,
        ) -> dict[str, object]:
            self.dispatched.append(task)
            return {"message": "all good"}

    parser = DummyParser()
    dispatcher = MessageDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("run", "exit"))
    out = capsys.readouterr().out
    assert "SUCCESS: all good" in out
    assert dispatcher.dispatched


def test_confirmation(monkeypatch, capsys):
    parser = DummyParser(confirm=True)
    dispatcher = DummyDispatcher()
    prompts = iter_inputs("do it", "exit")
    confirms = iter(["n"])

    def _confirm(_: str, default: bool | None = None) -> bool:
        return next(confirms) == "y"

    chat_loop(
        parser,
        dispatcher,
        prompt_fn=prompts,
        confirm_fn=_confirm,
    )
    out = capsys.readouterr().out
    assert "Cancelled" in out
    assert not dispatcher.dispatched


def test_chained_dispatch(capsys):
    class ChainParser(DummyParser):
        def parse(  # type: ignore[override]
            self, prompt: str
        ) -> list[dict[str, object]]:
            return [{"command": "a"}, {"command": "b"}]

    parser = ChainParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("multi", "exit"))
    out = capsys.readouterr().out
    assert "Step 1" in out and "Step 2" in out
    assert out.count("SUCCESS: ok") == 2
    assert len(dispatcher.dispatched) == 2


def test_hide_traceback(capsys):
    class ErrorParser(DummyParser):
        def parse(self, prompt: str) -> dict[str, object]:
            raise ValueError("boom")

    parser = ErrorParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("go", "exit"))
    captured = capsys.readouterr()
    assert "boom" in captured.out
    assert "Traceback" not in captured.err


def test_debug_shows_traceback(capsys):
    class ErrorParser(DummyParser):
        def parse(self, prompt: str) -> dict[str, object]:
            raise ValueError("boom")

    parser = ErrorParser()
    dispatcher = DummyDispatcher()
    chat_loop(
        parser,
        dispatcher,
        debug=True,
        prompt_fn=iter_inputs("go", "exit"),
    )
    captured = capsys.readouterr()
    assert "Traceback" in captured.err


def test_unknown_command_prints_message(capsys):
    class FailDispatcher(DummyDispatcher):
        def dispatch(  # type: ignore[override]
            self,
            task: object,
        ) -> dict[str, object]:
            return {
                "ok": False,
                "message": "Unknown command, type `help` to see options.",
            }

    parser = DummyParser()
    dispatcher = FailDispatcher()
    chat_loop(
        parser,
        dispatcher,
        prompt_fn=iter_inputs("run", "exit"),
    )
    out = capsys.readouterr().out
    assert "Unknown command, type `help` to see options." in out
    assert "SUCCESS" not in out


def test_unknown_text_provides_guidance(capsys):
    class NoCommandParser(DummyParser):
        def parse(  # type: ignore[override]
            self,
            prompt: str,
        ) -> dict[str, object]:
            return {}

    parser = NoCommandParser()
    dispatcher = DummyDispatcher()
    chat_loop(
        parser,
        dispatcher,
        prompt_fn=iter_inputs("unknown text", "exit"),
    )
    out = capsys.readouterr().out
    assert "Unknown command" in out
    assert "do foo" in out
    assert not dispatcher.dispatched


def test_failed_dispatch_prints_message(capsys):
    class FailDispatcher(DummyDispatcher):
        def dispatch(  # type: ignore[override]
            self,
            task: object,
        ) -> dict[str, object]:
            return {"ok": False, "message": "bad"}

    parser = DummyParser()
    dispatcher = FailDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("go", "exit"))
    out = capsys.readouterr().out
    assert "bad" in out
    assert "SUCCESS" not in out


def test_pipeline_prompt_dispatches_without_help(capsys):
    dispatcher = DummyDispatcher()
    prompts = iter_inputs(
        "Hey, can you run the full pipeline for NVDA?",
        "exit",
    )

    def _confirm(_: str, default: bool | None = None) -> bool:
        return True

    chat_loop(nl_parser, dispatcher, prompt_fn=prompts, confirm_fn=_confirm)
    out = capsys.readouterr().out

    assert dispatcher.dispatched
    task = dispatcher.dispatched[0]
    assert getattr(task, "command", None) == "pipeline.run_daily"
    assert "Available commands" not in out
