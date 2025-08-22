from sentimental_cap_predictor.agent.sandbox import run_subprocess, safe_shell


def test_run_subprocess_echo():
    proc = run_subprocess(["echo", "hi"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hi"


def test_safe_shell_allows_echo():
    res = safe_shell("echo hello")
    assert res.ok
    assert res.message.strip() == "hello"


def test_safe_shell_rejects_unknown_binary():
    res = safe_shell("sleep 0")
    assert not res.ok
    assert "not allowed" in res.message


def test_safe_shell_rejects_semicolons():
    res = safe_shell("echo hi; ls")
    assert not res.ok
    assert "not allowed" in res.message or "pipes and redirects" in res.message
