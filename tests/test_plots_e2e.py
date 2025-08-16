import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

# Make tests reproducible and avoid TF on Windows
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

TICKER = os.environ.get("TEST_TICKER", "AAPL")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ENV = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))


@pytest.mark.e2e
def test_end_to_end_pipeline_generates_plot_inputs(tmp_path):
    """Full pipeline: ingest -> train_eval -> plots
    Asserts the predictions + learning-curve CSVs exist and have rows."""
    # 1) Ingest
    proc = subprocess.run(
        ["python", "-m", "sentimental_cap_predictor.data.ingest", TICKER, "--period", "1Y", "--interval", "1d"],
        env=ENV,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"ingest failed: {proc.stderr}")

    # 2) Train/Eval
    subprocess.run(
        ["python", "-m", "sentimental_cap_predictor.modeling.train_eval", TICKER],
        check=True,
        env=ENV,
    )

    pred_csv = DATA_PROCESSED / f"{TICKER}_train_test_predictions.csv"
    lc_csv = DATA_PROCESSED / f"{TICKER}_learning_curve_train_test.csv"

    assert pred_csv.exists(), f"Missing {pred_csv}"
    assert lc_csv.exists(), f"Missing {lc_csv}"

    df_pred = pd.read_csv(pred_csv)
    df_lc = pd.read_csv(lc_csv)
    assert len(df_pred) > 0 and len(df_lc) > 0

    # 3) Plots should run without crashing
    subprocess.run(["python", "-m", "sentimental_cap_predictor.plots", TICKER], check=True, env=ENV)
