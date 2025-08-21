from __future__ import annotations

import json
import os
from typing import List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai is optional
    OpenAI = None

from .idea_schema import Idea

_DEFAULT_SYSTEM_PROMPT = (
    "You are a financial research assistant generating concise quantitative"
    " trading ideas. Return ideas as JSON."
)


def generate_ideas(topic: str, *, model: str = "gpt-4o-mini", n: int = 3) -> List[Idea]:
    """Generate research ideas from a text description using an LLM.

    Parameters
    ----------
    topic:
        Short text describing the market, asset class, or research question.
    model:
        Name of the chat model. The default assumes an OpenAI-compatible API.
    n:
        Number of ideas to request from the model.

    Returns
    -------
    list of :class:`Idea`
        Generated idea objects.
    """

    if OpenAI is None:  # pragma: no cover - defensive import check
        raise ImportError("openai package is required to generate ideas")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_prompt = (
        f"Propose {n} new quantitative trading ideas about: {topic}. "
        "Respond in JSON list where each item has fields 'name', 'description', and 'params'."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on model
        raise ValueError("Model response was not valid JSON") from exc

    return [Idea(**item) for item in raw]
