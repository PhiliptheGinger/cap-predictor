
# Command Line Interface

## Strategy Optimizer

Run the moving-average strategy optimizer using Typer:

```bash
python -m sentimental_cap_predictor.trader_utils.strategy_optimizer optimize data/processed/NVDA_prices.csv --iterations 250 --seed 42
```

The CSV must contain two columns:

- `date` – trading date in YYYY-MM-DD format
- `close` – closing price for the instrument

Example above runs optimizer on an NVDA CSV.

## Data Ingestion

Fetch price history and create optimizer-ready CSV:

```bash
python -m sentimental_cap_predictor.data.ingest NVDA --period 1Y --interval 1d
```

This saves:
- `data/raw/NVDA_prices.parquet` with columns `date, open, high, low, close, adj_close, volume`
- `data/processed/NVDA_prices.csv` containing only `date` and `close`

## Baseline Train/Eval

Run simple benchmark models and write evaluation CSVs:

```bash
python -m sentimental_cap_predictor.modeling.train_eval NVDA
```

This writes `data/processed/NVDA_train_test_predictions.csv` with columns:

- `date` – trading date in YYYY-MM-DD format
- `actual` – true target values
- `predicted` – model predictions
