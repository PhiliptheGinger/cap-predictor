# flake8: noqa
import json

from pydantic import BaseModel

from sentimental_cap_predictor.llm_core.agent.loop import AgentLoop
from sentimental_cap_predictor.llm_core.agent.tool_registry import (
    ToolSpec,
    register_tool,
)
from sentimental_cap_predictor.llm_core.agent.tools import web_search
from tools import file_io, python_exec


def test_search_read_transcript(monkeypatch):
    def fake_search(query, top_k=5):
        return [{"title": "A", "snippet": "B", "url": "http://example.com"}]

    monkeypatch.setattr(web_search, "search_web", fake_search)

    class ReadUrlInput(BaseModel):
        url: str

    class ReadUrlOutput(BaseModel):
        text: str

    def _read_handler(payload: ReadUrlInput) -> ReadUrlOutput:
        return ReadUrlOutput(text="article text")

    try:
        register_tool(
            ToolSpec(
                name="read_url",
                input_model=ReadUrlInput,
                output_model=ReadUrlOutput,
                handler=_read_handler,
            )
        )
    except ValueError:
        pass

    outputs = [
        'First CMD: {"name": "search.web", "input": {"query": "foo"}}',  # noqa: E501
        ('Second CMD: {"name": "read_url", "input": {"url": "http://example.com"}}'),  # noqa: E501
        "Final",
    ]

    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return outputs.pop(0)

    loop = AgentLoop(fake_llm, max_steps=3)
    result = loop.run("Task")

    assert result == "Final"
    search_obs = json.dumps(
        {"results": [{"title": "A", "snippet": "B", "url": "http://example.com"}]},  # noqa: E501
        separators=(",", ":"),
    )
    read_obs = json.dumps({"text": "article text"}, separators=(",", ":"))
    assert prompts == [
        "Task",
        f"Task\nObservation: {search_obs}",
        f"Task\nObservation: {search_obs}\nObservation: {read_obs}",
    ]


def test_file_write_python_run_transcript(monkeypatch):
    monkeypatch.setattr(
        file_io, "file_write", lambda path, content: f"agent_work/{path}"
    )
    monkeypatch.setattr(
        python_exec,
        "python_run",
        lambda *, code=None, path=None, timeout=10: {
            "stdout": "ok",
            "stderr": "",
            "paths": [],
        },
    )

    class PyRunInput(BaseModel):
        code: str | None = None
        path: str | None = None

    class PyRunOutput(BaseModel):
        stdout: str
        stderr: str
        paths: list[str]

    def _run_handler(payload: PyRunInput) -> PyRunOutput:
        return PyRunOutput(
            **python_exec.python_run(code=payload.code, path=payload.path)
        )

    try:
        register_tool(
            ToolSpec(
                name="python.run",
                input_model=PyRunInput,
                output_model=PyRunOutput,
                handler=_run_handler,
            )
        )
    except ValueError:
        pass

    outputs = [
        (
            'First CMD: {"name": "file.write", "input": {"path": '
            '"script.py", "content": "print(1)"}}'
        ),
        'Second CMD: {"name": "python.run", "input": {"path": "script.py"}}',
        "Done",
    ]

    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return outputs.pop(0)

    loop = AgentLoop(fake_llm, max_steps=3)
    result = loop.run("Task")

    assert result == "Done"
    write_obs = json.dumps({"path": "agent_work/script.py"}, separators=(",", ":"))  # noqa: E501
    run_obs = json.dumps(
        {"stdout": "ok", "stderr": "", "paths": []}, separators=(",", ":")
    )
    assert prompts == [
        "Task",
        f"Task\nObservation: {write_obs}",
        f"Task\nObservation: {write_obs}\nObservation: {run_obs}",
    ]
