# CAP Predictor User Manual

## Environment Setup
- Copy the sample environment file and fill in API keys or paths:
  ```bash
  cp .env.example .env
  ```
- Install dependencies in editable mode with development extras:
  ```bash
  pip install -e .[dev]
  ```

## Data Ingestion and Sentiment Analysis
- Download price and news data for a ticker:
  ```bash
  python -m sentimental_cap_predictor.dataset TICKER 1Y
  ```
  The `period` argument is positional and defaults to `max` if omitted.
- Generate plots or other visualizations:
  ```bash
  python -m sentimental_cap_predictor.plots TICKER
  ```
- Run the sentiment analysis module on collected news:
  ```bash
  python -m sentimental_cap_predictor.modeling.sentiment_analysis <NEWS_PATH>
  ```

## Model Training, Evaluation, and Daily Pipeline
- Train baseline models and write evaluation metrics:
  ```bash
  python -m sentimental_cap_predictor.modeling.train_eval TICKER
  ```
- Visualize results or build trading strategies using the provided utilities.
- Execute the end‑to‑end daily workflow:
  ```bash
  python -m sentimental_cap_predictor.flows.daily_pipeline run TICKER
  ```
  This pipeline downloads data, trains the model, searches for a strategy, and produces a summary report.

## Testing Guidance
- Run the test suite with `pytest`:
  ```bash
  pytest
  ```
- To avoid network calls, enable offline mode:
  ```bash
  OFFLINE_TEST=1 TEST_TICKER=AAPL pytest
  ```
  The `OFFLINE_TEST` flag makes tests use cached CSVs under `tests/data`.

## Additional Resources
- [Command Line Interface](cli.md)
- [Data Bundle Format](data_bundle.md)
- [Scheduling](scheduling.md)
- [Research Hooks](research.md)
