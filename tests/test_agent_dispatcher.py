from sentimental_cap_predictor.agent import command_registry
from sentimental_cap_predictor.agent.dispatcher import DispatchResult, dispatch


def test_dispatch_sys_status():
    result = dispatch({"command": "sys.status"})
    assert isinstance(result, DispatchResult)
    assert result.ok
    assert "python" in result.metrics and "platform" in result.metrics


def test_dispatch_missing_param():
    res = dispatch({"command": "file.read"})
    assert not res.ok
    assert "path" in res.message.lower()


def test_dispatch_custom_command(monkeypatch):
    def handler() -> dict[str, object]:
        return {
            "message": "done",
            "metrics": {"acc": 1},
            "artifacts": ["out.txt"],
        }

    cmd = command_registry.Command(
        name="custom",
        handler=handler,
        summary="",
        params_schema={},
    )
    from sentimental_cap_predictor.agent import dispatcher as dispatcher_module

    monkeypatch.setattr(
        dispatcher_module,
        "get_registry",
        lambda: {"custom": cmd},
    )
    res = dispatcher_module.dispatch({"command": "custom"})
    assert res.ok
    assert res.message == "done"
    assert res.metrics == {"acc": 1}
    assert res.artifacts == ["out.txt"]
