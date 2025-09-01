import subprocess

from sentimental_cap_predictor import chatbot_frontend as cf


def test_handle_command_routes_to_gdelt(monkeypatch):
    called = {}

    def fake_fetch(query):  # noqa: ANN001
        called["query"] = query
        return "Headline"

    monkeypatch.setattr(cf, "fetch_first_gdelt_article", fake_fetch)

    text = cf.handle_command(
        "curl https://api.gdeltproject.org/api/v2/doc/doc?query=NVDA"
    )
    assert called["query"] == "NVDA"
    assert text == "Headline"


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
