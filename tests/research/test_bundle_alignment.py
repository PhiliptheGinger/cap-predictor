import pandas as pd

from sentimental_cap_predictor.data.loader import align_daily, align_pit_fundamentals


def test_align_daily_forward_fill_limit():
    index = pd.date_range('2020-01-01', periods=35, freq='D')
    df = pd.DataFrame({'score': [1.0]}, index=[index[0]])
    aligned = align_daily(df, index)
    assert aligned.loc[index[1], 'score'] == 1.0
    assert aligned.loc[index[30], 'score'] == 1.0
    assert pd.isna(aligned.loc[index[31], 'score'])


def test_align_pit_fundamentals_forward_fill():
    index = pd.date_range('2020-01-01', periods=5, freq='D')
    fundamentals = pd.DataFrame({
        'date': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02')],
        'field': ['metric', 'metric'],
        'value': [10.0, 20.0],
        'asof_ts': [pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-04')],
    })
    aligned = align_pit_fundamentals(fundamentals, index)
    assert pd.isna(aligned.loc['2020-01-02', 'metric'])
    assert aligned.loc['2020-01-03', 'metric'] == 10.0
    assert aligned.loc['2020-01-04', 'metric'] == 20.0
