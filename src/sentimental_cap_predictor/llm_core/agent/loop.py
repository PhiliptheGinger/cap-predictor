from __future__ import annotations

import json
import logging
import time
from typing import Callable, List

from .tool_registry import get_tool

QUESTION_KEYWORDS = (
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
)

logger = logging.getLogger("audit")


class AgentLoop:
    """Simple controller implementing a Thought→CMD→Observation loop."""

    def __init__(
        self,
        llm: Callable[[str], str],
        max_steps: int = 5,
        timeout: float = 30.0,
        max_prompt_tokens: int = 4096,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._timeout = timeout
        self._max_prompt_tokens = max_prompt_tokens

    def run(self, prompt: str) -> str:
        """Execute the agent loop starting from ``prompt``.

        Parameters
        ----------
        prompt:
            Initial text sent to the model.

        Returns
        -------
        str
            Final answer when no ``CMD:`` block is present.

        Raises
        ------
        TimeoutError
            If the loop runs longer than ``timeout`` seconds.
        RuntimeError
            If ``max_steps`` is reached without producing a final answer.
        """

        observations: List[str] = []
        start = time.time()

        prompt = self._maybe_augment_prompt(prompt)

        for _ in range(self._max_steps):
            if time.time() - start > self._timeout:
                raise TimeoutError("Agent loop timed out")

            model_input = self._build_prompt(prompt, observations)
            model_input = self._truncate_tokens(model_input)
            model_output = self._llm(model_input)

            cmd = self._extract_cmd(model_output)
            if cmd is None:
                return model_output.strip()

            observation = self._dispatch(cmd)
            observations.append(observation)

        raise RuntimeError("Agent loop exceeded maximum steps")

    @staticmethod
    def _build_prompt(prompt: str, observations: List[str]) -> str:
        if not observations:
            return prompt
        obs_text = "\n".join(f"Observation: {o}" for o in observations)
        return f"{prompt}\n{obs_text}"

    def _maybe_augment_prompt(self, prompt: str) -> str:
        """Attach relevant memory snippets to ``prompt`` when appropriate."""

        if not self._looks_like_question(prompt):
            return prompt
        try:  # pragma: no cover - memory is optional
            from tools.memory import memory_query
        except Exception:
            return prompt
        try:
            hits = memory_query(prompt)
        except Exception:
            return prompt
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {})
            text = meta.get("text")
            source = meta.get("source") or meta.get("url") or hit.get("id")
            if not text:
                continue
            lines.append(f"[{idx}] {text} (source: {source})")
        if not lines:
            return prompt
        snippet = "\n".join(lines)
        return f"{prompt}\n\nSources:\n{snippet}"

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lower = text.lower()
        return any(word in lower for word in QUESTION_KEYWORDS)

    def _truncate_tokens(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) <= self._max_prompt_tokens:
            return text
        return " ".join(tokens[: self._max_prompt_tokens])

    @staticmethod
    def _extract_cmd(text: str) -> dict | None:
        if "CMD:" not in text:
            return None
        cmd_text = text.split("CMD:", 1)[1].strip()
        try:
            return json.loads(cmd_text)
        except json.JSONDecodeError as exc:  # invalid JSON
            return {"error": f"CMD parse error: {exc}"}

    @staticmethod
    def _dispatch(cmd: dict) -> str:
        if "error" in cmd:
            return cmd["error"]
        name = cmd.get("name")
        if not name:
            return "Missing tool name"
        try:
            spec = get_tool(name)
        except KeyError:
            return f"Unknown tool '{name}'"

        input_data = cmd.get("input", {})
        logger.info("CMD %s input=%s", name, input_data)
        input_model = spec.input_model.model_validate(input_data)
        result = spec.handler(input_model)
        if not isinstance(result, spec.output_model):
            result = spec.output_model.model_validate(result)
        logger.info("RESULT %s output=%s", name, result.model_dump())
        return result.model_dump_json()


__all__ = ["AgentLoop"]
