
"""Baseline model training and evaluation CLI."""
from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import os

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from sentimental_cap_predictor.features.builder import build_features
from sentimental_cap_predictor.prep.pipeline import add_returns, add_tech_indicators

try:  # optional dependency
    import mlflow
except Exception:  # pragma: no cover - mlflow optional
    mlflow = None

try:  # optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost optional
    XGBClassifier = None


app = typer.Typer(help="Train baseline models and materialize evaluation CSVs")


@app.command()
def main(ticker: str) -> None:
    """Train simple models and write prediction / learning curve CSVs."""

    use_mlflow = mlflow is not None and os.environ.get("MLFLOW_DISABLED") != "1"
    run_ctx = mlflow.start_run() if use_mlflow else nullcontext()

    with run_ctx:
        raw_path = Path("data/raw") / f"{ticker}_prices.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw price parquet at {raw_path}")
        df = pd.read_parquet(raw_path)

        # Compute returns for MAR ratio alignment
        df_ret = add_returns(df)
        df_ret = add_tech_indicators(df_ret)
        df_ret = df_ret.dropna().reset_index(drop=True)
        returns_series = df_ret["ret_1d"].shift(-1).iloc[:-1].reset_index(drop=True)

        X, y, dates = build_features(df, ticker=ticker)
        if len(returns_series) != len(y):
            raise ValueError("Return series and labels misaligned")
        if len(X) < 30:
            raise ValueError("Not enough data for training")

        split_idx = int(len(X) * 0.7)
        gap = 5
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx + gap :], y[split_idx + gap :]
        test_dates = dates.iloc[split_idx + gap :].reset_index(drop=True)
        test_returns = returns_series.iloc[split_idx + gap :].reset_index(drop=True)

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

        precision = precision_score(y_test, best_pred, zero_division=0)
        recall = recall_score(y_test, best_pred, zero_division=0)
        f1 = f1_score(y_test, best_pred, zero_division=0)
        if hasattr(best_model, "predict_proba"):
            probas = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probas)
        else:  # pragma: no cover - all models currently support predict_proba
            roc_auc = float("nan")

        positions = np.where(best_pred > 0, 1, -1)
        strategy_returns = positions * test_returns.to_numpy()
        equity_curve = (1 + pd.Series(strategy_returns)).cumprod()
        if len(equity_curve) > 1:
            years = len(equity_curve) / 252
            cagr = equity_curve.iloc[-1] ** (1 / years) - 1
            running_max = equity_curve.cummax()
            drawdown = (equity_curve / running_max) - 1
            max_drawdown = float(drawdown.min())
            mar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        else:
            mar_ratio = np.nan

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame(
            {
                "date": test_dates,
                "actual": y_test,
                "predicted": best_pred,
            }
        )
        pred_path = processed_dir / f"{ticker}_train_test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info("wrote %s", pred_path)
        if use_mlflow:
            mlflow.log_artifact(str(pred_path))

        metrics = {
            "accuracy": best_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "mar_ratio": mar_ratio,
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_path = processed_dir / f"{ticker}_train_test_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("wrote %s", metrics_path)
        if use_mlflow:
            mlflow.log_artifact(str(metrics_path))

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
        if use_mlflow:
            mlflow.log_artifact(str(lc_path))

        if use_mlflow:
            mlflow.log_param("model", best_name)
            mlflow.log_params(best_model.get_params())
            mlflow.log_metrics(metrics)


if __name__ == "__main__":  # pragma: no cover
    app()
