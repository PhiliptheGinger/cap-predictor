"""FastAPI application exposing endpoints for chat, model metrics,
asset performance and decision traceability.

This skeleton implements the UI concept described in the documentation.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CAP Predictor UI")


class ChatMessage(BaseModel):
    """Incoming message for the chatbot."""

    message: str


class ChatResponse(BaseModel):
    """Response returned by the chatbot."""

    id: str
    response: str


# Simple in-memory store to keep conversation state during the session.
_chat_history: list[ChatResponse] = []


@app.post("/chat", response_model=ChatResponse)
def chat(msg: ChatMessage) -> ChatResponse:
    """Return a placeholder response for a chat message.

    In a full implementation this would call an LLM to generate a reply.
    """

    resp = ChatResponse(id=str(uuid4()), response=f"Echo: {msg.message}")
    _chat_history.append(resp)
    return resp


@app.get("/metrics/{model_type}")
def get_metrics(model_type: str) -> dict[str, float | str]:
    """Return placeholder metrics for the requested model type."""

    return {"model": model_type, "accuracy": 0.0, "latency": 0.0}


@app.get("/assets/{asset_id}/performance")
def asset_performance(asset_id: str) -> dict[str, list[float] | str]:
    """Return placeholder asset-level performance data."""

    return {"asset": asset_id, "predictions": [], "actuals": []}


@app.get("/trace/{prediction_id}")
def trace(prediction_id: str) -> dict[str, str | list[str]]:
    """Return placeholder decision trace for a prediction."""

    return {"prediction_id": prediction_id, "trace": []}
