"""Demo: research and summarize a URL using agent tools."""
from __future__ import annotations
import importlib, pathlib, types, sys

BASE = pathlib.Path(__file__).resolve().parents[1] / "src" / "sentimental_cap_predictor" / "llm_core" / "agent"

# Stub package hierarchy to satisfy relative imports
pkg_root = types.ModuleType("sentimental_cap_predictor")
sys.modules["sentimental_cap_predictor"] = pkg_root
monitor_pkg = types.ModuleType("sentimental_cap_predictor.monitoring")
class RunLogger:  # minimal stub
    def log(self, **kwargs):
        pass
monitor_pkg.RunLogger = RunLogger
sys.modules["sentimental_cap_predictor.monitoring"] = monitor_pkg
llm_pkg = types.ModuleType("sentimental_cap_predictor.llm_core")
sys.modules["sentimental_cap_predictor.llm_core"] = llm_pkg
agent_pkg = types.ModuleType("sentimental_cap_predictor.llm_core.agent")
agent_pkg.__path__ = [str(BASE)]
sys.modules["sentimental_cap_predictor.llm_core.agent"] = agent_pkg

loop_mod = importlib.import_module("sentimental_cap_predictor.llm_core.agent.loop")
AgentLoop = loop_mod.AgentLoop
reg_mod = importlib.import_module("sentimental_cap_predictor.llm_core.agent.tool_registry")
ToolSpec = reg_mod.ToolSpec
register_tool = reg_mod.register_tool

from pydantic import BaseModel


class ReadUrlInput(BaseModel):
    url: str


class ReadUrlOutput(BaseModel):
    text: str


def _read_url(payload: ReadUrlInput) -> ReadUrlOutput:
    return ReadUrlOutput(text="Example Domain is for use in illustrative examples.")


try:  # pragma: no cover - registration is optional
    register_tool(
        ToolSpec(
            name="read.url",
            input_model=ReadUrlInput,
            output_model=ReadUrlOutput,
            handler=_read_url,
        )
    )
except ValueError:
    pass


RESPONSES = [
    'CMD: {"tool": "read.url", "input": {"url": "http://example.com"}}',
    "The page describes the Example Domain, a site for demonstrations.",
]


def _fake_llm(_prompt: str) -> str:
    return RESPONSES.pop(0)


def main() -> None:
    loop = AgentLoop(_fake_llm, max_steps=2)
    result = loop.run("Summarize http://example.com")
    print(result)


if __name__ == "__main__":
    main()
