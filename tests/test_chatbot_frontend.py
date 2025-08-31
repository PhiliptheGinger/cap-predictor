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
