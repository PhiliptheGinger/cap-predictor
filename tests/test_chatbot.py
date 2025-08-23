import types

import pytest

from sentimental_cap_predictor.chatbot import chat_loop


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


@pytest.mark.parametrize("trigger", ["help", "?"])
def test_help_lists_registry(trigger, capsys):
    parser = DummyParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs(trigger, "exit"))
    out = capsys.readouterr().out
    assert "do foo" in out and "foo bar" in out


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
        def dispatch(self, task: object) -> dict[str, object]:  # type: ignore[override]
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
        def dispatch(self, task: object) -> dict[str, object]:  # type: ignore[override]  # noqa: E501
            return {
                "ok": False,
                "message": "Unknown command, type `help` to see options.",
            }

    parser = DummyParser()
    dispatcher = FailDispatcher()
    chat_loop(
        parser,
        dispatcher,
        prompt_fn=iter_inputs("hey, what can you do?", "exit"),
    )
    out = capsys.readouterr().out
    assert "Unknown command, type `help` to see options." in out
    assert "SUCCESS" not in out


def test_unknown_text_provides_guidance(capsys):
    class NoCommandParser(DummyParser):
        def parse(self, prompt: str) -> dict[str, object]:  # type: ignore[override]
            return {}

    parser = NoCommandParser()
    dispatcher = DummyDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("unknown text", "exit"))
    out = capsys.readouterr().out
    assert "Unknown command" in out
    assert "do foo" in out
    assert not dispatcher.dispatched


def test_failed_dispatch_prints_message(capsys):
    class FailDispatcher(DummyDispatcher):
        def dispatch(self, task: object) -> dict[str, object]:  # type: ignore[override]
            return {"ok": False, "message": "bad"}

    parser = DummyParser()
    dispatcher = FailDispatcher()
    chat_loop(parser, dispatcher, prompt_fn=iter_inputs("go", "exit"))
    out = capsys.readouterr().out
    assert "bad" in out
    assert "SUCCESS" not in out
