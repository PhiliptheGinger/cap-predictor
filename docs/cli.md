
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

Additionally, `data/processed/NVDA_train_test_metrics.csv` is generated
containing overall evaluation metrics:

- `accuracy`, `precision`, `recall`, `f1`, `roc_auc` and `mar_ratio`.

Runs are logged to MLflow by default. Set environment variable
`MLFLOW_DISABLED=1` to skip MLflow logging.

## Daily Pipeline

Run the full daily workflow:

```bash
python -m sentimental_cap_predictor.flows.daily_pipeline run NVDA
```

This downloads data, preprocesses it, trains the model, searches for an
optimized moving‑average strategy, performs a backtest and writes a JSON summary
to `data/processed/NVDA_daily_summary.json`.

### Scheduling

Use cron to execute the pipeline every weekday at 6am:

```cron
0 6 * * 1-5 python -m sentimental_cap_predictor.flows.daily_pipeline run NVDA >> logs/daily.log 2>&1
```

Make sure the virtual environment is activated or provide the full path to the
Python interpreter in the cron entry.

## Idea Generation

Produce candidate trading ideas with a local language model and optionally
write them to a JSON file:

```bash
python -m sentimental_cap_predictor.scheduler ideas:generate "interest rate regimes" --output ideas.json
```

This command can be scheduled with cron in the same way as the daily pipeline.

## Chatbot

Interact with a local Mistral-powered assistant from the terminal. The
chatbot queries both a *main* and an *experimental* model for every question
and explains which answer was chosen:

```bash
python -m sentimental_cap_predictor.chatbot \
    --main-model mistralai/Mistral-7B-v0.1 \
    --experimental-model mistralai/Mistral-7B-Instruct-v0.2
```

## GitHub Repository Data

Utilities that pull information from the GitHub API can optionally use a
personal access token to increase rate limits.  Set the token via the
`GITHUB_TOKEN` environment variable or supply it on the command line with
`--token <TOKEN>` when available.

## Additional Resources

- [User Manual](user_manual.md) – step-by-step setup and workflows
- [Documentation Index](index.md) – links to all project guides
