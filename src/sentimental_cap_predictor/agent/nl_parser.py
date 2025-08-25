from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from .command_registry import get_registry

# Registry ------------------------------------------------------------------
registry = get_registry()

# Canonicalization -----------------------------------------------------------
# Map conversational synonyms to a canonical phrase so downstream regular
# expressions only need to account for the standardized form. This keeps the
# regexes manageable while still allowing flexible user phrasing.
SYNONYM_MAP = {
    "full pipeline": "daily pipeline",
    "entire pipeline": "daily pipeline",
    "whole pipeline": "daily pipeline",
}


@dataclass
class Intent:
    """Represents the result of parsing a natural language request.

    Attributes
    ----------
    command:
        Name of the command to execute. ``None`` if no command could be
        inferred.
    params:
        Mapping of parameter names to values extracted from the text.
    requires_confirmation:
        Whether the command should be confirmed by the user before execution.
    confidence:
        Heuristic confidence score between 0 and 1.
    """

    command: str | None
    params: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    confidence: float = 0.0

    # Backwards compatibility -------------------------------------------------
    # ``action`` was the original field name. Provide a property so existing
    # code that still accesses ``intent.action`` continues to work without
    # modification.
    @property
    def action(self) -> str | None:  # pragma: no cover - legacy support
        return self.command

    @action.setter  # pragma: no cover - legacy support
    def action(self, value: str | None) -> None:
        self.command = value


def parse(
    text: str, llm: Callable[[str], "Intent"] | None = None
) -> Intent | List[Intent]:
    """Parse ``text`` into one or more :class:`Intent` objects.

    Besides semicolons and the phrase ``"and then"``, the parser also accepts
    simple ``"and"`` as a command separator when the subsequent text starts
    with a known command keyword. This enables prompts such as ``"fetch SPY and
    train model SPY"`` to be interpreted as two distinct actions.
    """

    # Normalize common "(period/interval)" syntax to "period interval" so the
    # ingestion regex can parse it.
    text = re.sub(
        r"\((\d+[a-z]+)/(\d+[a-z]+)\)",
        r" \1 \2 ",
        text,
        flags=re.IGNORECASE,
    )

    # Split chained commands. The lookahead ensures that plain "and" inside
    # parameters (e.g. "compare 1 and 2") are not treated as separators.
    splitter = re.compile(
        r"\s*(?:;|\band\s+then\b|"
        r"\band\b(?=\s*(?:ingest|download|fetch|train|retrain|optimize|"
        r"compare|promote|list|show|run|generate|ideas?|shell|tests|pytest|"
        r"pipeline|system|status)))\s*",
        flags=re.IGNORECASE,
    )
    parts = splitter.split(text)
    intents = [_parse_single(part, llm) for part in parts if part.strip()]
    return intents[0] if len(intents) == 1 else intents


# ---------------------------------------------------------------------------
# Internal single-command parser
# ---------------------------------------------------------------------------


def _parse_single(
    text: str,
    llm: Callable[[str], Intent] | None = None,
) -> Intent:
    """Parse a single command ``text`` into an :class:`Intent`.

    The parser implements a collection of regular-expression and keyword
    heuristics for the most common commands. If no rule matches, an optional
    ``llm`` callable can be used as a secondary parser. If that also fails, a
    simple fallback heuristic is applied.
    """

    original = text.strip()

    # Replace known synonyms with their canonical equivalents prior to
    # matching so that the subsequent regexes only need to consider a single
    # phrase for each concept.
    for phrase, canonical in SYNONYM_MAP.items():
        original = re.sub(
            rf"\b{re.escape(phrase)}\b",
            canonical,
            original,
            flags=re.IGNORECASE,
        )

    lowered = original.lower()

    # data.ingest ------------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:ingest|download|fetch)\s+(?P<ticker>[A-Za-z0-9_]+)"
        r"(?:\s+(?P<period>\d+[a-z]+))?"
        r"(?:\s+(?P<interval>\d+[a-z]+))?",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        params = {k: v for k, v in m.groupdict().items() if v}
        return Intent("data.ingest", params, confidence=0.9)

    # model.train_eval -------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:train(?:\s+model)?|retrain(?:\s+the\s+model)?|"
        r"model\.train_eval)\s+"
        r"(?:for\s+)?(?P<ticker>[A-Za-z0-9_]+)",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        return Intent(
            "model.train_eval",
            {"ticker": m.group("ticker")},
            confidence=0.9,
        )

    # strategy.optimize ------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:optimize|strategy\.optimize)\s+(?P<csv_path>\S+)"
        r"(?:\s+(?P<iterations>\d+))?"
        r"(?:\s+(?P<seed>\d+))?"
        r"(?:\s+(?P<lambda_drawdown>[0-9.]+))?",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        params = {k: v for k, v in m.groupdict().items() if v}
        if "iterations" in params:
            params["iterations"] = int(params["iterations"])
        if "seed" in params:
            params["seed"] = int(params["seed"])
        if "lambda_drawdown" in params:
            params["lambda_drawdown"] = float(params["lambda_drawdown"])
        return Intent("strategy.optimize", params, confidence=0.9)

    # ideas.generate ---------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:ideas?|ideas\.generate|gen ideas)\s+(?P<topic>\w+)"
        r"(?:\s+(?P<model_id>\w+))?"
        r"(?:\s+(?P<n>\d+))?",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        params = {k: v for k, v in m.groupdict().items() if v}
        if "n" in params:
            params["n"] = int(params["n"])
        return Intent("ideas.generate", params, confidence=0.9)

    # experiments.compare ----------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:compare|experiments\.compare)\s+"
        r"(?P<first>\d+)\s+(?P<second>\d+)",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        params = {
            "first": int(m.group("first")),
            "second": int(m.group("second")),
        }
        return Intent("experiments.compare", params, confidence=0.9)

    # file.read --------------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:file\.read|read|cat)\s+(?P<path>.+)",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        return Intent(
            "file.read",
            {"path": m.group("path").strip()},
            confidence=0.9,
        )

    # model.promote ----------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:model\.promote|promote)\s+(?P<src>\S+)\s+(?P<dst>\S+)",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        params = {"src": m.group("src"), "dst": m.group("dst")}
        return Intent(
            "model.promote",
            params,
            requires_confirmation=True,
            confidence=0.8,
        )

    # tests.run --------------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:tests(?:\.run)?|run tests|pytest)\b(?:\s+(?P<args>.*))?",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        args_str = m.group("args")
        args = shlex.split(args_str) if args_str else None
        return Intent("tests.run", {"args": args}, confidence=0.9)

    # shell.run --------------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:shell\.run|!|shell|bash|sh)\s+(?P<cmd>.+)",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        return Intent(
            "shell.run",
            {"cmd": m.group("cmd")},
            requires_confirmation=True,
            confidence=0.7,
        )

    # pipeline.run_daily -----------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:pipeline\.run_daily|run (?:the )?daily pipeline)\s+"
        r"(?P<ticker>\w+)(?:\s+(?P<period>\S+))?(?:\s+(?P<interval>\S+))?",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        params = {k: v for k, v in m.groupdict().items() if v}
        return Intent(
            "pipeline.run_daily",
            params,
            requires_confirmation=True,
            confidence=0.8,
        )

    # experiments.list -------------------------------------------------------
    if re.match(r"(?:^|\b)(?:experiments\.list|list experiments)$", lowered):
        return Intent("experiments.list", {}, confidence=0.9)

    # experiments.show -------------------------------------------------------
    m = re.match(
        r"(?:^|\b)(?:experiments\.show|show experiment)\s+(?P<run_id>\d+)",
        original,
        flags=re.IGNORECASE,
    )
    if m:
        return Intent(
            "experiments.show",
            {"run_id": int(m.group("run_id"))},
            confidence=0.9,
        )

    # sys.status -------------------------------------------------------------
    if re.match(r"(?:^|\b)(?:sys\.status|system status|status)$", lowered):
        return Intent("sys.status", {}, confidence=0.9)

    # Fallback ---------------------------------------------------------------
    if llm is not None:
        try:
            intent = llm(original)
            if isinstance(intent, Intent):
                return intent
        except Exception:
            pass
    return _fallback_heuristic(original)


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def _fallback_heuristic(text: str) -> Intent:
    """Very small heuristic used when no rule matches.

    If ``text`` looks like a path to an existing file, we assume the user wants
    to read it. Otherwise we return an empty intent.
    """

    candidate = Path(text)
    if candidate.exists():
        return Intent("file.read", {"path": text}, confidence=0.2)
    return Intent(command=None, params={"text": text}, confidence=0.0)


__all__ = ["Intent", "parse", "registry"]
