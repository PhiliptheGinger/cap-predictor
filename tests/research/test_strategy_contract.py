import inspect
import re

import pandas as pd

from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.research.engine import SmaSentStrategy
from sentimental_cap_predictor.research.idea_schema import Idea


def test_signal_alignment_and_bounds_and_no_shift():
    index = pd.date_range('2020-01-01', periods=5, freq='D')
    prices = pd.DataFrame({'close': range(1, 6)}, index=index)
    sentiment = pd.DataFrame({'score': [0, 1, 0.5, 1, 0]}, index=index)
    data = DataBundle(prices=prices, sentiment=sentiment).validate()
    idea = Idea(name='sma', params={'ma_window': 2, 'sent_thr': 0.3})
    strat = SmaSentStrategy()

    signals = strat.generate_signals(data, idea)
    assert signals.index.equals(prices.index)
    assert ((signals >= 0) & (signals <= 1)).all()

    src = inspect.getsource(SmaSentStrategy.generate_signals)
    assert re.search(r"shift\s*\(\s*-1\s*\)", src) is None

