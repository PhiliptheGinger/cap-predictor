from __future__ import annotations

from pathlib import Path
from typing import Dict

from .io_types import Argument, DispatchDecision, NLUResult, Resolution
from .ontology import Ontology
from .policy import Policy
from .dispatcher import Dispatcher
from .reasoner import Reasoner
from .qwen_intent import QwenNLU

# ---------------------------------------------------------------------------
# Initialize components
# ---------------------------------------------------------------------------

_pkg_path = Path(__file__).resolve().parent
_ontology = Ontology(_pkg_path / "intents.yaml")
_engine = QwenNLU()
_policy = Policy(_ontology)
_dispatcher = Dispatcher()
_reasoner = Reasoner(_ontology)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(utterance: str, ctx: Dict) -> NLUResult:
    ctx["utterance"] = utterance
    nlu = _engine.predict(utterance)
    if nlu.intent:
        required = _ontology.required_slots(nlu.intent)
        nlu.missing_slots = [s for s in required if s not in nlu.slots]
    return nlu


def resolve(nlu: NLUResult, ctx: Dict) -> Resolution:
    return _policy.resolve(nlu, ctx)


def dispatch(res: Resolution, ctx: Dict) -> DispatchDecision:
    decision = _dispatcher.dispatch(res, ctx)
    ctx["last_decision"] = decision
    ctx["last_intent"] = res.intent
    return decision


def explain(decision: DispatchDecision, nlu: NLUResult, ctx: Dict) -> Argument:
    return _reasoner.explain(decision, nlu, ctx)

