import pandas as pd
import pytest

from sentimental_cap_predictor.data_bundle import DataBundle


def test_research_bundle_validate_rejects_misalignment():
    idx = pd.date_range('2020-01-01', periods=2, freq='D')
    prices = pd.DataFrame({'close': [1, 2]}, index=idx)
    bad_sent = pd.DataFrame({'score': [0.1, 0.2]}, index=idx[::-1])
    with pytest.raises(ValueError):
        DataBundle(prices=prices, sentiment=bad_sent).validate()
