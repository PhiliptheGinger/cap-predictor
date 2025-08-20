import pandas as pd
import pytest

from sentimental_cap_predictor.data_bundle import DataBundle


def test_get_series_and_convenience_access():
    index = pd.date_range('2024-01-01', periods=3, freq='D')
    prices = pd.DataFrame({
        ('AAPL', 'close'): [1, 2, 3],
        ('MSFT', 'close'): [2, 3, 4],
    }, index=index)
    features = pd.DataFrame({('AAPL', 'feat'): [0.1, 0.2, 0.3]}, index=index)
    sentiment = pd.DataFrame({('AAPL', 'score'): [0.5, 0.6, 0.7]}, index=index)

    bundle = DataBundle(prices=prices, features=features, sentiment=sentiment).validate()

    pd.testing.assert_series_equal(bundle.get_series('AAPL', 'close'), prices[('AAPL', 'close')])
    pd.testing.assert_series_equal(bundle.get_feature('AAPL', 'feat'), features[('AAPL', 'feat')])
    pd.testing.assert_series_equal(bundle.get_sentiment('AAPL', 'score'), sentiment[('AAPL', 'score')])


def test_publication_timestamps_enforce_no_lookahead():
    index = pd.date_range('2024-01-01', periods=2, freq='D')
    prices = pd.DataFrame({('AAPL', 'close'): [1, 2]}, index=index)
    pub_ok = pd.DataFrame({('AAPL', 'close'): index - pd.Timedelta(hours=1)}, index=index)

    # Should validate successfully
    DataBundle(prices=prices, publication_times={'prices': pub_ok}).validate()

    pub_bad = pd.DataFrame({('AAPL', 'close'): index + pd.Timedelta(days=1)}, index=index)
    with pytest.raises(ValueError):
        DataBundle(prices=prices, publication_times={'prices': pub_bad}).validate()
