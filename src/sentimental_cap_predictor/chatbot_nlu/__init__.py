from __future__ import annotations

from pathlib import Path
from typing import Dict

from .io_types import Argument, DispatchDecision, NLUResult, Resolution
from .nlu import NLUEngine
from .ontology import Ontology
from .policy import Policy
from .dispatcher import Dispatcher
from .reasoner import Reasoner

# ---------------------------------------------------------------------------
# Initialize components
# ---------------------------------------------------------------------------

_pkg_path = Path(__file__).resolve().parent
_ontology = Ontology(_pkg_path / "intents.yaml")
_engine = NLUEngine.from_files(
    _pkg_path / "intents.yaml", _pkg_path / "examples" / "seed_utterances.jsonl"
)
_policy = Policy(_ontology)
_dispatcher = Dispatcher()
_reasoner = Reasoner(_ontology)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(utterance: str, ctx: Dict) -> NLUResult:
    ctx["utterance"] = utterance
    return _engine.parse(utterance)


def resolve(nlu: NLUResult, ctx: Dict) -> Resolution:
    return _policy.resolve(nlu, ctx)


def dispatch(res: Resolution, ctx: Dict) -> DispatchDecision:
    decision = _dispatcher.dispatch(res, ctx)
    ctx["last_decision"] = decision
    ctx["last_intent"] = res.intent
    return decision


def explain(decision: DispatchDecision, nlu: NLUResult, ctx: Dict) -> Argument:
    return _reasoner.explain(decision, nlu, ctx)

