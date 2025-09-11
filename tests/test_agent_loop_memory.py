import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

# flake8: noqa


def _load_agent_loop():
    root = Path(__file__).resolve().parents[1]
    pkg = SimpleNamespace(__path__=[])
    sys.modules.setdefault("sentimental_cap_predictor", pkg)

    llm_pkg = SimpleNamespace(__path__=[])
    sys.modules.setdefault("sentimental_cap_predictor.llm_core", llm_pkg)
    pkg.llm_core = llm_pkg

    agent_pkg = SimpleNamespace(__path__=[])
    sys.modules.setdefault("sentimental_cap_predictor.llm_core.agent", agent_pkg)  # noqa: E501
    llm_pkg.agent = agent_pkg

    tool_registry_mod = SimpleNamespace(get_tool=lambda name: None)
    sys.modules["sentimental_cap_predictor.llm_core.agent.tool_registry"] = (
        tool_registry_mod
    )
    agent_pkg.tool_registry = tool_registry_mod

    loop_path = (
        root / "src" / "sentimental_cap_predictor" / "llm_core" / "agent" / "loop.py"
    )  # noqa: E501
    spec = importlib.util.spec_from_file_location(
        "sentimental_cap_predictor.llm_core.agent.loop", loop_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sentimental_cap_predictor.llm_core.agent.loop"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.AgentLoop


def test_agentloop_queries_memory_and_truncates(monkeypatch):
    prompts = []

    def fake_llm(text: str) -> str:
        prompts.append(text)
        return "done"

    calls = []

    def fake_memory_query(query: str, top_k: int = 5):
        calls.append(query)
        return [
            {
                "id": "1",
                "metadata": {
                    "text": ("one two three four five six seven eight nine ten eleven"),  # noqa: E501
                    "source": "A",
                },
            }
        ]

    monkeypatch.setattr("tools.memory.memory_query", fake_memory_query)

    AgentLoop = _load_agent_loop()
    loop = AgentLoop(fake_llm, max_steps=1, max_prompt_tokens=10)
    loop.run("Who?")

    assert calls, "memory.query was not called"
    assert len(prompts) == 1
    sent_prompt = prompts[0]
    assert "Sources:" in sent_prompt
    assert "[1]" in sent_prompt
    assert "eleven" not in sent_prompt  # truncated
    assert len(sent_prompt.split()) <= 10
