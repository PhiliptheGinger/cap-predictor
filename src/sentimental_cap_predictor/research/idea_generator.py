"""Idea generation using a local Hugging Face model.

This module intentionally avoids any network API calls by running a
Hugging Face ``text-generation`` pipeline locally. The model ID defaults to
the ``MAIN_MODEL`` environment variable or ``Qwen/Qwen2-7B-Instruct`` if it is
unset. The default can be overridden to use any compatible base model that is
available on disk or through the Hugging Face hub. The model is prompted to
return a JSON list of ideas which are converted into
:class:`~sentimental_cap_predictor.research.idea_schema.Idea` instances.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

from .idea_schema import Idea

load_dotenv()


_DEFAULT_PROMPT = (
    "You are a financial research assistant generating concise quantitative "
    "trading ideas. Return ideas as JSON."
)


@lru_cache(maxsize=1)
def _get_pipeline(model_id: str):
    """Return a text-generation pipeline for ``model_id``.

    The heavy ``transformers`` import is done lazily here so that merely
    importing this module does not require the optional deep learning
    frameworks (e.g. TensorFlow) that ``transformers`` tries to load by
    default.  This keeps test environments lightweight and avoids import-time
    errors when those libraries are unavailable.

    The pipeline is explicitly configured to use **PyTorch** as the backend and
    to disable TensorFlow and Flax.  This prevents ``transformers`` from
    attempting to import those libraries which can trigger DLL errors on
    Windows systems where TensorFlow is not installed or not supported.
    """

    import os

    # Explicitly disable TensorFlow and Flax before importing ``transformers``.
    # ``USE_TF``/``USE_FLAX`` force ``is_tf_available`` and ``is_flax_available``
    # to return ``False`` even if the packages are present in the environment.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    from transformers import pipeline

    # ``framework="pt"`` guarantees the pipeline uses PyTorch only.
    return pipeline(
        "text-generation", model=model_id, tokenizer=model_id, framework="pt"
    )


def generate_ideas(
    topic: str,
    *,
    model_id: str = os.getenv("MAIN_MODEL", "Qwen/Qwen2-7B-Instruct"),
    n: int = 3,
) -> List[Idea]:
    """Generate research ideas from ``topic`` using a local language model.

    Parameters
    ----------
    topic:
        Short text describing the market, asset class, or research question.
    model_id:
        Hugging Face model identifier or local path to load. Defaults to the
        ``MAIN_MODEL`` environment variable or ``Qwen/Qwen2-7B-Instruct`` if
        unset.
    n:
        Number of ideas to request from the model.

    Returns
    -------
    list of :class:`Idea`
        Generated idea objects parsed from the model's JSON output.
    """

    generator = _get_pipeline(model_id)
    user_prompt = (
        f"{_DEFAULT_PROMPT}\n\nPropose {n} new quantitative trading ideas "
        f"about: {topic}. Respond in JSON list where each item has fields "
        "'name', 'description', and 'params'."
    )
    result = generator(
        user_prompt, max_new_tokens=512, do_sample=True, temperature=0.2
    )[0]["generated_text"]
    try:
        raw = json.loads(result)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on model
        raise ValueError("Model response was not valid JSON") from exc

    return [Idea(**item) for item in raw]


__all__ = ["generate_ideas"]
