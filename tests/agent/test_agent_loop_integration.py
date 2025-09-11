# flake8: noqa
import json

from sentimental_cap_predictor.llm_core.agent.loop import AgentLoop
from sentimental_cap_predictor.llm_core.agent.tools import web_search
from sentimental_cap_predictor.llm_core.llm_providers import deepseek


def test_agent_loop_dispatch(monkeypatch):
    def fake_search(query, top_k=5):
        return [{"title": "T", "snippet": "S", "url": "http://x"}]

    monkeypatch.setattr(web_search, "search_web", fake_search)

    outputs = [
        'Step CMD: {"name": "search.web", "input": {"query": "news"}}',
        "Final answer",
    ]

    prompts: list[str] = []

    def fake_chat(prompt: str) -> str:
        prompts.append(prompt)
        return outputs.pop(0)

    dummy = object.__new__(deepseek.DeepSeekProvider)
    dummy.chat = lambda messages, **kwargs: fake_chat(messages[0]["content"])  # noqa: E501

    loop = AgentLoop(lambda text: dummy.chat([{"role": "user", "content": text}]))  # noqa: E501
    result = loop.run("Hello?")

    assert result == "Final answer"
    expected_obs = json.dumps(
        {"results": [{"title": "T", "snippet": "S", "url": "http://x"}]},
        separators=(",", ":"),
    )
    assert prompts == ["Hello?", f"Hello?\nObservation: {expected_obs}"]
