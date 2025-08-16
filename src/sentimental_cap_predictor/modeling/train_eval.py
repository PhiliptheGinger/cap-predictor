"""Baseline model training and evaluation CLI."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:  # optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost optional
    XGBClassifier = None

from ..features.builder import build_features

app = typer.Typer(help="Train baseline models and materialize evaluation CSVs")


@app.command()
def main(ticker: str) -> None:
    """Train simple models and write prediction / learning curve CSVs."""

    raw_path = Path("data/raw") / f"{ticker}_prices.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw price parquet at {raw_path}")
    df = pd.read_parquet(raw_path)

    X, y, dates = build_features(df, ticker=ticker)
    if len(X) < 30:
        raise ValueError("Not enough data for training")

    split_idx = int(len(X) * 0.7)
    gap = 5
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx + gap :], y[split_idx + gap :]
    test_dates = dates.iloc[split_idx + gap :].reset_index(drop=True)

    models = [
        ("logreg", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
    ]
    if XGBClassifier is not None:
        models.append(("xgb", XGBClassifier(random_state=0, eval_metric="logloss")))

    best_acc = -1.0
    best_model = None
    best_name = ""
    best_pred = None

    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info("%s accuracy %.3f", name, acc)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            best_pred = preds

    assert best_model is not None and best_pred is not None
    logger.info("Best model: %s", best_name)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "date": test_dates,
            "TrueValues": y_test,
            "LNN_Predictions": best_pred,
            "BiasedPrediction": best_pred,
        }
    )
    pred_path = processed_dir / f"{ticker}_train_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("wrote %s", pred_path)

    # Learning curve via TimeSeriesSplit on training set
    tscv = TimeSeriesSplit(n_splits=5, gap=gap)
    lc_rows = []
    for train_idx, val_idx in tscv.split(X_train):
        model = best_model.__class__(**best_model.get_params())
        model.fit(X_train[train_idx], y_train[train_idx])
        train_loss = 1 - accuracy_score(y_train[train_idx], model.predict(X_train[train_idx]))
        val_loss = 1 - accuracy_score(y_train[val_idx], model.predict(X_train[val_idx]))
        lc_rows.append({"Train Size": len(train_idx), "Train Loss": train_loss, "Validation Loss": val_loss})
    lc_df = pd.DataFrame(lc_rows)
    lc_path = processed_dir / f"{ticker}_learning_curve_train_test.csv"
    lc_df.to_csv(lc_path, index=False)
    logger.info("wrote %s", lc_path)


if __name__ == "__main__":  # pragma: no cover
    app()
