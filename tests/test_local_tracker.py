from research.tracking import LocalTracker, Metric, Param


def test_params_and_metrics_queryable(tmp_path):
    tracker = LocalTracker(root=tmp_path)
    run_id = tracker.start_run({"lr": 0.1})
    tracker.log_metrics({"loss": 0.5})

    with tracker.SessionLocal() as session:
        rows = (
            session.query(Metric, Param)
            .join(Param, Metric.run_id == Param.run_id)
            .filter(Metric.run_id == run_id)
            .all()
        )

    assert len(rows) == 1
    metric, param = rows[0]
    assert metric.key == "loss" and metric.value == 0.5
    assert param.key == "lr" and param.value == "0.1"
