from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from .io_types import DispatchDecision, Resolution


@dataclass
class Dispatcher:
    """Map intents to project functions. Uses simple in-module stubs."""

    registry: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.registry.update(
            {
                "pipeline.run_daily": self._run_daily,
                "pipeline.run_now": self._run_now,
                "data.ingest": self._ingest,
                "model.train_eval": self._train_eval,
                "plots.make_report": self._make_report,
                "explain.decision": self._explain_last,
                "help.show_options": self._help,
            }
        )

    # ------------------------------------------------------------------
    def dispatch(self, res: Resolution, ctx: Dict) -> DispatchDecision:
        if res.action_needed != "DISPATCH" or res.intent is None:
            return DispatchDecision(action=None, args={}, executed=False, result=None)
        fn = self.registry.get(res.intent)
        if not fn:
            return DispatchDecision(action=res.intent, args=res.slots, executed=False, result=None)
        result = fn(**res.slots)
        return DispatchDecision(action=res.intent, args=res.slots, executed=True, result=result)

    # ------------------------------------------------------------------
    def _run_daily(self) -> Dict[str, Any]:
        return {"summary": "Scheduled daily pipeline"}

    def _run_now(self) -> Dict[str, Any]:
        return {"summary": "Pipeline executed"}

    def _ingest(self, tickers, period, interval) -> Dict[str, Any]:
        return {"summary": f"Ingested {tickers} for {period} at {interval}"}

    def _train_eval(self, ticker, split=None, seed=None) -> Dict[str, Any]:
        return {"summary": f"Trained model for {ticker}"}

    def _make_report(self, ticker, range=None) -> Dict[str, Any]:
        return {"summary": f"Generated report for {ticker}"}

    def _explain_last(self) -> Dict[str, Any]:
        return {"summary": "Explained last action"}

    def _help(self) -> Dict[str, Any]:
        return {"summary": "Showed options"}
