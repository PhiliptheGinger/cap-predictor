import pytest
import requests

from tools import read_url


def test_read_url_success(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.headers = {"Content-Type": "text/html"}
            self.text = "<html><title>Hi</title><body>Hello</body></html>"
            self.url = "http://example.com"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers, timeout: DummyResponse(),
    )
    result = read_url.read_url("http://example.com")
    assert "Hello" in result["text"]
    assert result["meta"]["url"] == "http://example.com"
    assert result["meta"].get("title") == "Hi"


def test_read_url_validation_error(monkeypatch):
    def fake_get(url, headers, timeout):
        raise requests.exceptions.InvalidURL("bad url")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(requests.exceptions.InvalidURL):
        read_url.read_url("not-a-url")


def test_read_url_failure(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.headers = {"Content-Type": "text/html"}
            self.text = ""
            self.url = "http://example.com"

        def raise_for_status(self):
            raise requests.HTTPError("404")

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers, timeout: DummyResponse(),
    )
    with pytest.raises(requests.HTTPError):
        read_url.read_url("http://example.com")
