import pandas as pd
import pytest

from research.sandbox import SandboxError, run_strategy_source
from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.research.idea_schema import Idea


def test_forbidden_import_raises_sandbox_error():
    index = pd.date_range("2020-01-01", periods=1, freq="D")
    prices = pd.DataFrame({"close": [1.0]}, index=index)
    data = DataBundle(prices=prices).validate()
    idea = Idea(name="x")
    source = "import os\nstrategy = None"
    with pytest.raises(SandboxError):
        run_strategy_source(source, data, idea)


def test_infinite_loop_terminates():
    index = pd.date_range("2020-01-01", periods=1, freq="D")
    prices = pd.DataFrame({"close": [1.0]}, index=index)
    data = DataBundle(prices=prices).validate()
    idea = Idea(name="x")
    source = "\n".join(
        [
            "def strategy(data, idea, ctx):",
            "    while True:",
            "        pass",
        ]
    )
    with pytest.raises(SandboxError):
        run_strategy_source(source, data, idea, max_ops=1000)
