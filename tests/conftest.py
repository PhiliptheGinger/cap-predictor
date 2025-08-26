import sys
from pathlib import Path

# Ensure the src directory is importable for tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import pytest

from sentimental_cap_predictor.chatbot_nlu.io_types import NLUResult
import sentimental_cap_predictor.chatbot_nlu as chatbot_nlu


@pytest.fixture(autouse=True)
def mock_qwen(monkeypatch):
    """Provide a deterministic stand-in for the Qwen model during tests."""

    def fake_predict(utterance: str) -> NLUResult:
        u = utterance.lower()
        if "daily pipeline" in u:
            return NLUResult(intent="pipeline.run_daily", scores=None, slots={})
        if "pipeline now" in u:
            return NLUResult(intent="pipeline.run_now", scores=None, slots={})
        if "ingest" in u:
            tickers = []
            if "nvda" in u:
                tickers.append("NVDA")
            if "aapl" in u:
                tickers.append("AAPL")
            slots = {"tickers": tickers or ["NVDA"], "period": "5d", "interval": "1h"}
            return NLUResult(intent="data.ingest", scores=None, slots=slots)
        if "help" in u:
            return NLUResult(intent="help.show_options", scores=None, slots={})
        return NLUResult(intent="help.show_options", scores=None, slots={})

    monkeypatch.setattr(chatbot_nlu._engine, "predict", fake_predict)
