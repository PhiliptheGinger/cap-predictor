import json
from pathlib import Path

from sentimental_cap_predictor.llm_core.agent.loop import AgentLoop
from sentimental_cap_predictor.monitoring import RunLogger


def test_agent_loop_writes_run_log(tmp_path, monkeypatch):
    rl = RunLogger(base_dir=tmp_path)

    responses = ['CMD: {"tool": "dummy"}', "done"]
    loop = AgentLoop(lambda _: responses.pop(0), max_steps=2)
    loop._run_logger = rl

    monkeypatch.setattr(AgentLoop, "_dispatch", lambda self, cmd: "ok")

    loop.run("hello world")

    files = list(Path(tmp_path).glob("*.jsonl"))
    assert files, "run log not created"
    content = files[0].read_text().strip().splitlines()
    record = json.loads(content[0])
    assert record["tool_name"] == "dummy"
    assert record["tokens_in"] > 0
    assert record["tokens_out"] > 0
