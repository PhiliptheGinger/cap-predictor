"""Demo: write and execute a Python function via agent tools."""
from __future__ import annotations
import importlib, pathlib, types, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "src" / "sentimental_cap_predictor" / "llm_core" / "agent"
sys.path.append(str(ROOT))  # allow importing tools

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

from tools import file_io, python_exec  # noqa: F401 - ensure tool registration

RESPONSES = [
    'CMD: {"name": "file.write", "input": {"path": "demo.py", "content": "def add(a,b):\n    return a+b\nprint(add(2,3))"}}',
    'CMD: {"name": "python.run", "input": {"path": "demo.py"}}',
    "Finished running the function.",
]


def _fake_llm(_prompt: str) -> str:
    return RESPONSES.pop(0)


def main() -> None:
    loop = AgentLoop(_fake_llm, max_steps=3)
    result = loop.run("Write a Python function and execute it")
    print(result)


if __name__ == "__main__":
    main()
