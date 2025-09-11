import subprocess

import pytest

from tools import python_exec


def test_python_run_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(python_exec.shutil, "which", lambda name: None)
    result = python_exec.python_run(code="print('hi')")
    assert result["stdout"].strip() == "hi"
    assert result["stderr"] == ""


def test_python_run_validation_error():
    with pytest.raises(ValueError):
        python_exec.python_run(code="print(1)", path="script.py")


def test_python_run_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="py", timeout=10)

    monkeypatch.setattr(python_exec.subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        python_exec.python_run(code="print('hi')")
