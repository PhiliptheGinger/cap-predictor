from __future__ import annotations

import json
import logging
import time
from typing import Callable, List

from .tool_registry import get_tool

logger = logging.getLogger("audit")


class AgentLoop:
    """Simple controller implementing a Thought→CMD→Observation loop."""

    def __init__(
        self,
        llm: Callable[[str], str],
        max_steps: int = 5,
        timeout: float = 30.0,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._timeout = timeout

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
        for _ in range(self._max_steps):
            if time.time() - start > self._timeout:
                raise TimeoutError("Agent loop timed out")

            model_input = self._build_prompt(prompt, observations)
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
