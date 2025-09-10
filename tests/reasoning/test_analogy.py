from sentimental_cap_predictor.reasoning.analogy import (
    best_metaphor,
    map_roles,
)


def test_map_roles_basic() -> None:
    source = {"hero": "Frodo", "villain": "Sauron"}
    target = {"hero": "Luke", "villain": "Vader", "mentor": "Yoda"}
    assert map_roles(source, target) == {"Frodo": "Luke", "Sauron": "Vader"}


def test_best_metaphor_known() -> None:
    label, rationale = best_metaphor("brain")
    assert label == "computer"
    assert "information" in rationale


def test_best_metaphor_unknown() -> None:
    label, rationale = best_metaphor("unknown")
    assert label == ""
    assert "No metaphor" in rationale
