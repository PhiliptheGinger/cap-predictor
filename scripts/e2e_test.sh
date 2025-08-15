#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/e2e_test.sh AAPL
TICKER="${1:-AAPL}"
export TRANSFORMERS_NO_TF=1

# 1) Prices → raw parquet + processed CSV
python -m sentimental_cap_predictor.data.ingest "$TICKER" --period 1Y --interval 1d

# 2) Train/Eval → writes processed predictions & learning curve CSVs
python -m sentimental_cap_predictor.modeling.train_eval "$TICKER"

# 3) Optimizer (optional) → writes best JSON
python -m sentimental_cap_predictor.trader_utils.strategy_optimizer optimize "data/processed/${TICKER}_prices.csv" --iterations 200 --seed 1 || true

# 4) Strategy signals (optional)
python -m sentimental_cap_predictor.strategies.moving_average gen \
  --prices "data/processed/${TICKER}_prices.csv" \
  --short 10 --long 30 \
  --out "data/processed/${TICKER}_signals.csv" || true

# 5) Backtest (optional)
python -m sentimental_cap_predictor.backtest.engine run \
  --prices "data/processed/${TICKER}_prices.csv" \
  --signals "data/processed/${TICKER}_signals.csv" \
  --commission-bps 10 --slippage-bps 10 --size 1.0 || true

# 6) Plots (expects train_eval CSVs)
python -m sentimental_cap_predictor.plots "$TICKER"

echo "E2E OK for ${TICKER}"
