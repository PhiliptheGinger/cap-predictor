from __future__ import annotations

"""FastAPI application exposing command dispatch endpoints."""

from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from sentimental_cap_predictor.agent import command_registry, dispatcher

app = FastAPI()


class RunRequest(BaseModel):
    """Request body for the run endpoint."""

    command: str
    params: Dict[str, Any] | None = None


@app.get("/commands")
def list_commands() -> Dict[str, Any]:
    """Return available command summaries and example payloads."""

    registry = command_registry.get_registry()
    commands: Dict[str, Any] = {}
    for name, cmd in registry.items():
        example_params = {k: f"<{v}>" for k, v in (cmd.params_schema or {}).items()}
        commands[name] = {
            "summary": cmd.summary,
            "example": {"command": name, "params": example_params},
        }
    return commands


@app.post("/run")
def run_command(request: RunRequest) -> Dict[str, Any]:
    """Execute the requested command via the dispatcher."""

    intent = {"command": request.command, "params": request.params or {}}
    result = dispatcher.dispatch(intent)
    return {
        "ok": result.ok,
        "message": result.message,
        "artifacts": result.artifacts,
        "metrics": result.metrics,
    }
