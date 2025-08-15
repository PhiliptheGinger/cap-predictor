"""Simplified training/evaluation CLI writing prediction and learning-curve CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

app = typer.Typer(help="Train baseline models and materialize evaluation CSVs")


@app.command()
def main(ticker: str) -> None:
    """Read processed prices and create stub prediction/learning-curve outputs."""
    processed_dir = Path("data/processed")
    price_path = processed_dir / f"{ticker}_prices.csv"
    if not price_path.exists():
        raise FileNotFoundError(f"Missing price csv at {price_path}")

    df = pd.read_csv(price_path)

    # minimal placeholder predictions
    preds = pd.DataFrame(
        {
            "date": df["date"],
            "TrueValues": df["close"],
            "LNN_Predictions": df["close"],
            "BiasedPrediction": df["close"],
        }
    )
    pred_path = processed_dir / f"{ticker}_train_test_predictions.csv"
    preds.to_csv(pred_path, index=False)
    logger.info("wrote %s", pred_path)

    lc = pd.DataFrame({"Train Size": [len(df)], "Train Loss": [0.0], "Validation Loss": [0.0]})
    lc_path = processed_dir / f"{ticker}_learning_curve_train_test.csv"
    lc.to_csv(lc_path, index=False)
    logger.info("wrote %s", lc_path)


if __name__ == "__main__":  # pragma: no cover
    app()
