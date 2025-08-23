from sentimental_cap_predictor.agent.dispatcher import dispatch
from sentimental_cap_predictor import experiment


def test_experiment_commands_surface_paths(tmp_path, monkeypatch):
    tracker = experiment.ExperimentTracker(db_path=tmp_path / "exp.db")
    art1 = tmp_path / "a.txt"
    art1.write_text("a")
    art2 = tmp_path / "b.txt"
    art2.write_text("b")
    run1 = tracker.log("code", {}, {"acc": 0.9}, {"a": str(art1)})
    run2 = tracker.log("code", {}, {"acc": 0.8}, {"b": str(art2)})

    monkeypatch.setattr(experiment, "ExperimentTracker", lambda: tracker)

    res_list = dispatch({"command": "experiments.list"})
    assert res_list.ok
    assert res_list.metrics[str(run1)]["acc"] == 0.9
    assert str(art1) in res_list.artifacts
    assert str(art2) in res_list.artifacts

    res_show = dispatch({"command": "experiments.show", "run_id": run1})
    assert res_show.ok
    assert res_show.metrics["acc"] == 0.9
    assert res_show.artifacts == [str(art1)]

    res_cmp = dispatch(
        {"command": "experiments.compare", "first": run1, "second": run2}
    )
    assert res_cmp.ok
    assert res_cmp.metrics["first"]["acc"] == 0.9
    assert res_cmp.metrics["second"]["acc"] == 0.8
    assert str(art1) in res_cmp.artifacts and str(art2) in res_cmp.artifacts
