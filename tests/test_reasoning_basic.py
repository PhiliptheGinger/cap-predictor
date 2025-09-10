from sentimental_cap_predictor.reasoning.analogy import map_roles
from sentimental_cap_predictor.reasoning.schemas import ContainerSchema
from sentimental_cap_predictor.reasoning.simulator import step


def test_container_schema_describe() -> None:
    container = ContainerSchema(name="box", capacity=10.0, volume=5.0)
    description = container.describe()
    assert description


def test_map_roles_simple() -> None:
    source = {"a": 1}
    target = {"a": 2}
    assert map_roles(source, target) == {1: 2}


def test_simulator_strong_force_upward_narration() -> None:
    state = {"trend": "up", "force": "strong"}
    _, narration = step(state)
    assert "up" in narration.lower()
