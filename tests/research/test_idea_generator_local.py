import importlib
import json
from dataclasses import asdict

from sentimental_cap_predictor.research import idea_generator


class _StubPipeline:
    def __call__(
        self,
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
    ):
        data = [{"name": "Idea", "description": "desc", "params": {"x": 1}}]
        return [{"generated_text": json.dumps(data)}]


def test_generate_ideas_parses_json(monkeypatch):
    monkeypatch.setattr(
        idea_generator, "_get_pipeline", lambda model_id: _StubPipeline()
    )
    ideas = idea_generator.generate_ideas("topic", model_id="dummy", n=1)
    assert [asdict(i) for i in ideas] == [
        {"name": "Idea", "description": "desc", "params": {"x": 1}}
    ]


def test_generate_ideas_uses_env_default(monkeypatch):
    called = {}

    def _stub(model_id):
        called["model"] = model_id
        return _StubPipeline()

    monkeypatch.setenv("MAIN_MODEL", "env-model")
    importlib.reload(idea_generator)
    monkeypatch.setattr(idea_generator, "_get_pipeline", _stub)
    idea_generator.generate_ideas("topic", n=1)
    assert called["model"] == "env-model"
