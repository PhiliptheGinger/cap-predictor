from sentimental_cap_predictor.experiment import ExperimentTracker


def test_log_and_query(tmp_path):
    db_path = tmp_path / "experiments.db"
    tracker = ExperimentTracker(db_path=db_path)

    code = "print('hi')"
    params = {"lr": 0.01}
    metrics = {"accuracy": 0.9}
    chart = tmp_path / "chart.png"
    chart.write_text("fake image")
    artifacts = {"chart": str(chart)}

    run_id = tracker.log(code, params, metrics, artifacts)

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["id"] == run_id

    run = tracker.get_run(run_id)
    assert run["params"]["lr"] == 0.01
    assert run["metrics"]["accuracy"] == 0.9

    tracker.add_artifact(run_id, "trades", tmp_path / "trades.log")
    artifacts_loaded = tracker.get_artifacts(run_id)
    assert "trades" in artifacts_loaded

    # Write trade log and verify load_artifact helper
    trade_log = artifacts_loaded["trades"]
    trade_log.write_text("some trades")
    assert tracker.load_artifact(run_id, "trades").strip() == "some trades"

