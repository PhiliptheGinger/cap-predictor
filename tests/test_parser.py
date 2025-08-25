import pytest

from sentimental_cap_predictor.agent.nl_parser import parse


@pytest.mark.parametrize(
    "text,command,params",
    [
        (
            "ingest SPY 1y 1d",
            "data.ingest",
            {"ticker": "SPY", "period": "1y", "interval": "1d"},
        ),
        ("train model AAPL", "model.train_eval", {"ticker": "AAPL"}),
        ("compare 1 2", "experiments.compare", {"first": 1, "second": 2}),
        ("system status", "sys.status", {}),
        (
            "Hey, can you run the full pipeline?",
            "pipeline.run_daily",
            {},
        ),
    ],
)
def test_regex_intent_mapping(
    text: str, command: str, params: dict[str, object]
) -> None:
    intent = parse(text)
    assert intent.command == command
    assert intent.params == params


def test_requires_confirmation() -> None:
    intent = parse("promote foo bar")
    assert intent.command == "model.promote"
    assert intent.requires_confirmation


@pytest.mark.parametrize(
    "text",
    [
        "ingest SPY 1y 1d; train model SPY",
        "ingest SPY 1y 1d and then train model SPY",
        "ingest SPY 1y 1d and train model SPY",
    ],
)
def test_chained_commands(text: str) -> None:
    intents = parse(text)
    assert isinstance(intents, list)
    assert [i.command for i in intents] == [
        "data.ingest",
        "model.train_eval",
    ]
