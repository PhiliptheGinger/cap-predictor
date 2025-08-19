
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

To run without network access, supply a CSV with the expected columns:

```bash
python -m sentimental_cap_predictor.data.ingest AAPL --offline-path tests/data/AAPL_prices.csv
```

When `--offline-path` is used, the file is read directly and no `yfinance` call is made.
