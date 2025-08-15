
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
