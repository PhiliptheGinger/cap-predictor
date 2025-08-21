import pandas as pd

from sentimental_cap_predictor import dataset


def test_check_for_nan_logs_warning(caplog):
    df = pd.DataFrame({"a": [1, None]})
    log_id = dataset.logger.add(caplog.handler, level="INFO")
    dataset.check_for_nan(df, "test")
    dataset.logger.remove(log_id)
    assert any(
        "Warning: Found 1 NaN values after test" in record.getMessage()
        for record in caplog.records
    )
