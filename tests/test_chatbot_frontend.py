import subprocess
import requests
import importlib.util
from pathlib import Path

# Import the module directly to avoid triggering package-level side effects
spec = importlib.util.spec_from_file_location(
    "chatbot_frontend",
    Path(__file__).resolve().parents[1]
    / "src"
    / "sentimental_cap_predictor"
    / "chatbot_frontend.py",
)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_fetch_first_gdelt_article(monkeypatch):
    payload = {
        "articles": [
            {"title": "Headline", "url": "http://example.com"}
        ]
    }

    def fake_get(url, params, timeout):  # noqa: ANN001
        assert params["query"] == "NVDA"
        return DummyResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    text = cf.fetch_first_gdelt_article("NVDA")
    assert text == "Headline - http://example.com"


def test_handle_command_routes_to_gdelt(monkeypatch):
    payload = {
        "articles": [
            {"title": "Headline", "url": "http://example.com"}
        ]
    }

    def fake_get(url, params, timeout):  # noqa: ANN001
        return DummyResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "Headline - http://example.com"


def test_handle_command_fallback(monkeypatch):
    def fake_get(url, params, timeout):  # noqa: ANN001
        return DummyResponse({"articles": []})

    monkeypatch.setattr(requests, "get", fake_get)

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert text == "No news found."


def test_handle_command_runs_shell(monkeypatch):
    class DummyCompleted:
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: DummyCompleted())
    out = cf.handle_command("echo hi")
    assert out == "ok"


def test_retry_on_malformed_output(monkeypatch):
    """Ensure malformed replies trigger a single retry with a reminder."""

    outputs = ["unexpected", "CMD: echo hi"]
    calls: list[str] = []

    class DummyProvider:
        def chat(self, history):  # noqa: ANN001
            calls.append(history[-1]["content"])
            return outputs.pop(0)

    import sys
    from types import SimpleNamespace

    dummy_module = SimpleNamespace(
        QwenLocalProvider=lambda *a, **k: DummyProvider(),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentimental_cap_predictor.llm_providers.qwen_local",
        dummy_module,
    )
    monkeypatch.setattr(
        "sentimental_cap_predictor.config_llm.get_llm_config",
        lambda: SimpleNamespace(model_path="", temperature=0.0),
    )

    user_inputs = iter(["hi", "quit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(user_inputs))

    executed = {}

    def fake_handle(cmd):  # noqa: ANN001
        executed["cmd"] = cmd
        return "done"

    monkeypatch.setattr(cf, "handle_command", fake_handle)
    cf.main()

    assert calls == [
        "hi",
        "Output invalid. Remember the CMD contract.",
    ]
    assert executed["cmd"] == "echo hi"
