from sentimental_cap_predictor.agent.sandbox import safe_shell


def test_safe_shell_whitelisted_binary():
    res = safe_shell("echo hello")
    assert res.ok
    assert res.message.strip() == "hello"


def test_safe_shell_rejects_semicolon():
    res = safe_shell("echo hi; ls")
    assert not res.ok


def test_safe_shell_rejects_double_ampersand():
    res = safe_shell("echo hi && ls")
    assert not res.ok


def test_safe_shell_rejects_pipe():
    res = safe_shell("echo hi | grep hi")
    assert not res.ok


def test_safe_shell_rejects_unknown_binary():
    res = safe_shell("foobar")
    assert not res.ok
