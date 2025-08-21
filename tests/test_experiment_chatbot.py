from sentimental_cap_predictor.experiment import (
    ExperimentTracker,
    _handle_question,
)


def test_handle_question_compare(tmp_path):
    tracker = ExperimentTracker(db_path=tmp_path / "experiments.db")
    run_a = tracker.log("code", {}, {"accuracy": 0.9}, {})
    run_b = tracker.log("code", {}, {"accuracy": 0.8}, {})
    msg = f"compare {run_a} {run_b} metric=accuracy"
    response = _handle_question(msg, tracker)
    assert str(run_a) in response
    assert "accuracy" in response
    assert "0.9" in response and "0.8" in response
