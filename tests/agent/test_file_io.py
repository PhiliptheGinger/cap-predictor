from pathlib import Path

import pytest

from tools import file_io


def test_file_write_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = file_io.file_write("out.txt", "hello")
    assert Path(path).read_text() == "hello"


def test_file_write_validation_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(file_io, "MAX_WRITE_BYTES", 1)
    with pytest.raises(ValueError):
        file_io.file_write("a.txt", "too long")


def test_file_write_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        file_io.file_write("../oops.txt", "hi")
