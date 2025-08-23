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
