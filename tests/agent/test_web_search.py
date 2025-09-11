import pytest
from pydantic import ValidationError

from sentimental_cap_predictor.llm_core.agent.tools import web_search


def test_search_web_success(monkeypatch):
    class DummyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results):
            assert query == "foo"
            assert max_results == 5
            return [
                {
                    "title": "Foo",
                    "body": "Bar",
                    "href": "http://example.com",
                }
            ]

    monkeypatch.setattr(web_search, "DDGS", DummyDDGS)
    results = web_search.search_web("foo")
    assert results == [
        {
            "title": "Foo",
            "snippet": "Bar",
            "url": "http://example.com",
        }
    ]


def test_search_web_validation_error():
    with pytest.raises(ValidationError):
        web_search.WebSearchInput.model_validate({"top_k": 1})


def test_search_web_failure(monkeypatch):
    monkeypatch.setattr(web_search, "DDGS", None)
    with pytest.raises(RuntimeError):
        web_search.search_web("foo")
