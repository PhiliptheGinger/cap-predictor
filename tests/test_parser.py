import pytest

from sentimental_cap_predictor.agent.nl_parser import parse


@pytest.mark.parametrize(
    "text,action,params",
    [
        (
            "ingest SPY 1y 1d",
            "data.ingest",
            {"ticker": "SPY", "period": "1y", "interval": "1d"},
        ),
        ("train model AAPL", "model.train_eval", {"ticker": "AAPL"}),
        ("compare 1 2", "experiments.compare", {"first": 1, "second": 2}),
        ("system status", "sys.status", {}),
    ],
)
def test_regex_intent_mapping(
    text: str, action: str, params: dict[str, object]
) -> None:
    intent = parse(text)
    assert intent.action == action
    assert intent.params == params


def test_requires_confirmation() -> None:
    intent = parse("promote foo bar")
    assert intent.action == "model.promote"
    assert intent.requires_confirmation
