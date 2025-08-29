import sys
import sys
import pandas as pd

from types import SimpleNamespace, ModuleType


def _stub_prices():
    dates = pd.date_range('2020-01-01', periods=4, freq='D')
    return pd.DataFrame({'date': dates, 'close': [1, 2, 3, 4]})


def test_ticker_logs_flag(monkeypatch, caplog, tmp_path):
    stub_module = ModuleType("sentimental_cap_predictor.model_training")
    stub_module.train_model = lambda train_df, random_state=None: None
    stub_module.predict_on_test_data = (
        lambda processed, model, test_df, sentiment_df: processed.assign(
            predicted=processed['close']
        )
    )
    monkeypatch.setitem(sys.modules, 'sentimental_cap_predictor.model_training', stub_module)

    from sentimental_cap_predictor.flows import daily_pipeline as dp

    # stub dependencies
    monkeypatch.setattr(dp.data_ingest, 'fetch_prices', lambda ticker, period='5y', interval='1d': _stub_prices())
    monkeypatch.setattr(dp.data_ingest, 'save_prices', lambda df, ticker: None)
    monkeypatch.setattr(dp.data_ingest, 'prices_to_csv_for_optimizer', lambda df, ticker: None)
    monkeypatch.setattr(dp, 'preprocess_price_data', lambda prices: (prices.set_index('date'), None))
    monkeypatch.setattr(dp.strat_opt, 'random_search', lambda series: SimpleNamespace(short_window=1, long_window=2, score=0.0, mean_return=0.0, mean_drawdown=0.0))
    monkeypatch.setattr(dp.strat_opt, 'moving_average_crossover', lambda series, s, l: 0.0)
    monkeypatch.setattr(dp, '_summary_path', lambda ticker: tmp_path / f"{ticker}_summary.json")

    log_id = dp.logger.add(caplog.handler, level="INFO")
    # Logs disabled by default
    monkeypatch.setattr(dp, 'ENABLE_TICKER_LOGS', False)
    dp.run('ABC')
    dp.logger.remove(log_id)
    assert 'Summary report written' not in caplog.text

    caplog.clear()
    log_id = dp.logger.add(caplog.handler, level="INFO")
    monkeypatch.setattr(dp, 'ENABLE_TICKER_LOGS', True)
    dp.run('ABC')
    dp.logger.remove(log_id)
    assert 'Summary report written' in caplog.text
