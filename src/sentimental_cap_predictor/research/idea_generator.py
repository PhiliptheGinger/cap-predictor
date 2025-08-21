"""Idea generation using a local Hugging Face model.

This module intentionally avoids any network API calls by running a
Hugging Face ``text-generation`` pipeline locally.  The default model id can
be overridden to use any compatible base model that is available on disk or
through the Hugging Face hub.  The model is prompted to return a JSON list of
ideas which are converted into :class:`~sentimental_cap_predictor.research.idea_schema.Idea`
instances.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List

from transformers import pipeline

from .idea_schema import Idea


_DEFAULT_PROMPT = (
    "You are a financial research assistant generating concise quantitative "
    "trading ideas. Return ideas as JSON."
)


@lru_cache(maxsize=1)
def _get_pipeline(model_id: str):
    """Return a text-generation pipeline for ``model_id``.

    The pipeline is cached to avoid re-loading weights when ``generate_ideas``
    is called multiple times within the same process.
    """

    return pipeline("text-generation", model=model_id, tokenizer=model_id)


def generate_ideas(
    topic: str, *, model_id: str = "mistralai/Mistral-7B-v0.1", n: int = 3
) -> List[Idea]:
    """Generate research ideas from ``topic`` using a local language model.

    Parameters
    ----------
    topic:
        Short text describing the market, asset class, or research question.
    model_id:
        Hugging Face model identifier or local path to load.
    n:
        Number of ideas to request from the model.

    Returns
    -------
    list of :class:`Idea`
        Generated idea objects parsed from the model's JSON output.
    """

    generator = _get_pipeline(model_id)
    user_prompt = (
        f"{_DEFAULT_PROMPT}\n\nPropose {n} new quantitative trading ideas about: {topic}. "
        "Respond in JSON list where each item has fields 'name', 'description', and 'params'."
    )
    result = generator(user_prompt, max_new_tokens=512, do_sample=True, temperature=0.2)[0][
        "generated_text"
    ]
    try:
        raw = json.loads(result)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on model
        raise ValueError("Model response was not valid JSON") from exc

    return [Idea(**item) for item in raw]


__all__ = ["generate_ideas"]

