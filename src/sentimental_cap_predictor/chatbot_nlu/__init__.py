"""Helpers for Qwen-powered intent parsing."""

from pathlib import Path

from . import qwen_intent
from .dispatcher import Dispatcher
from .io_types import Argument, DispatchDecision, NLUResult, Resolution
from .ontology import Ontology
from .policy import Policy
from .reasoner import Reasoner
from types import SimpleNamespace

_ONT_PATH = Path(__file__).with_name("intents.yaml")
_ONTOLOGY = Ontology(_ONT_PATH)
_POLICY = Policy(_ONTOLOGY)
_DISPATCHER = Dispatcher()
_REASONER = Reasoner(_ONTOLOGY)
_engine = SimpleNamespace(predict=qwen_intent.predict)


def parse(utterance: str, ctx: dict) -> NLUResult:
    data = _engine.predict(utterance)
    if isinstance(data, NLUResult):
        return data
    intent = data.get("intent")
    slots = data.get("slots", {}) or {}
    required = _ONTOLOGY.required_slots(intent) if intent else []
    missing = [s for s in required if s not in slots]
    return NLUResult(intent=intent, scores=None, slots=slots, missing_slots=missing)


def resolve(nlu: NLUResult, ctx: dict) -> Resolution:
    return _POLICY.resolve(nlu, ctx)


def dispatch(res: Resolution, ctx: dict) -> DispatchDecision:
    return _DISPATCHER.dispatch(res, ctx)


def explain(dec: DispatchDecision, nlu: NLUResult, ctx: dict) -> Argument:
    return _REASONER.explain(dec, nlu, ctx)


__all__ = [
    "parse",
    "resolve",
    "dispatch",
    "explain",
    "qwen_intent",
]
