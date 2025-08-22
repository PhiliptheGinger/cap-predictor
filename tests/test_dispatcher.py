from sentimental_cap_predictor.agent import command_registry
from sentimental_cap_predictor.agent import dispatcher as dispatcher_module
from sentimental_cap_predictor.agent.dispatcher import DispatchResult


def test_dispatch_uses_handler_and_structures_result(monkeypatch):
    calls: list[int] = []

    def handler(x: int) -> dict[str, object]:
        calls.append(x)
        return {
            "summary": "done",
            "metrics": {"x": x},
            "artifacts": ["out.txt"],
        }

    cmd = command_registry.Command(
        name="dummy",
        handler=handler,
        summary="",
        params_schema={"x": "int"},
    )
    monkeypatch.setattr(
        dispatcher_module,
        "get_registry",
        lambda: {"dummy": cmd},
    )

    res = dispatcher_module.dispatch({"command": "dummy", "x": 5})
    assert calls == [5]
    assert res == DispatchResult(
        ok=True, message="done", metrics={"x": 5}, artifacts=["out.txt"]
    )


def test_dispatch_validation_failure(monkeypatch):
    def handler(x: int) -> None:
        return None

    cmd = command_registry.Command(
        name="dummy",
        handler=handler,
        summary="",
        params_schema={"x": "int"},
    )
    monkeypatch.setattr(
        dispatcher_module,
        "get_registry",
        lambda: {"dummy": cmd},
    )

    res = dispatcher_module.dispatch({"command": "dummy", "x": "a"})
    assert not res.ok
    assert "x" in res.message
