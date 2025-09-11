"""Demo: chain search -> read -> memory -> code -> answer."""
from __future__ import annotations
import importlib, pathlib, types, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "src" / "sentimental_cap_predictor" / "llm_core" / "agent"
sys.path.append(str(ROOT))

pkg_root = types.ModuleType("sentimental_cap_predictor")
sys.modules["sentimental_cap_predictor"] = pkg_root
monitor_pkg = types.ModuleType("sentimental_cap_predictor.monitoring")
class RunLogger:
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

from pydantic import BaseModel, Field
from tools import python_exec  # noqa: F401 - ensure python.run registration


class SearchInput(BaseModel):
    query: str
    top_k: int = 5


class SearchOutput(BaseModel):
    results: list[dict[str, str]]


def _search(payload: SearchInput) -> SearchOutput:
    return SearchOutput(
        results=[{"title": "Example", "snippet": "Demo page", "url": "http://example.com"}]
    )


try:
    register_tool(
        ToolSpec(
            name="search.web",
            input_model=SearchInput,
            output_model=SearchOutput,
            handler=_search,
        )
    )
except ValueError:
    pass


class ReadInput(BaseModel):
    url: str


class ReadOutput(BaseModel):
    text: str


def _read(payload: ReadInput) -> ReadOutput:
    return ReadOutput(text="Example Domain text")


try:
    register_tool(
        ToolSpec(
            name="read.url",
            input_model=ReadInput,
            output_model=ReadOutput,
            handler=_read,
        )
    )
except ValueError:
    pass


class UpsertInput(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class UpsertOutput(BaseModel):
    success: bool


def _upsert(payload: UpsertInput) -> UpsertOutput:
    return UpsertOutput(success=True)


try:
    register_tool(
        ToolSpec(
            name="memory.upsert",
            input_model=UpsertInput,
            output_model=UpsertOutput,
            handler=_upsert,
        )
    )
except ValueError:
    pass


RESPONSES = [
    'CMD: {"tool": "search.web", "input": {"query": "demo"}}',
    'CMD: {"tool": "read.url", "input": {"url": "http://example.com"}}',
    'CMD: {"tool": "memory.upsert", "input": {"id": "ex", "text": "Example Domain text", "metadata": {"url": "http://example.com"}}}',
    'CMD: {"tool": "python.run", "input": {"code": "print(6*7)"}}',
    "Stored info and computed result 42.",
]


def _fake_llm(_prompt: str) -> str:
    return RESPONSES.pop(0)


def main() -> None:
    loop = AgentLoop(_fake_llm, max_steps=5)
    result = loop.run("Find info, store it, run code, and answer")
    print(result)


if __name__ == "__main__":
    main()
