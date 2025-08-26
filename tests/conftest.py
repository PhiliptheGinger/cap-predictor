import sys
from pathlib import Path

import pytest

# Ensure the src directory is on the Python path so tests can import the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentimental_cap_predictor.chatbot_nlu.qwen_intent import QwenNLU


@pytest.fixture(autouse=True)
def mock_qwen(monkeypatch):
    """Intercept calls to the Qwen API and return canned responses."""

    def _fake_call(self, utterance: str) -> str:
        text = utterance.lower()
        if "daily pipeline" in text:
            return '{"intent": "pipeline.run_daily", "slots": {}}'
        if "run the pipeline now" in text:
            return '{"intent": "pipeline.run_now", "slots": {}}'
        if "ingest" in text and "nvda" in text:
            return (
                '{"intent": "data.ingest", "slots": {"tickers": '
                '["NVDA", "AAPL"], "period": "5d", "interval": "1h"}}'
            )
        if "help me out" in text:
            return '{"intent": "help.show_options", "slots": {}}'
        # Anything else returns garbled output to trigger fallback
        return "not-json"

    monkeypatch.setattr(QwenNLU, "_call_model", _fake_call)

