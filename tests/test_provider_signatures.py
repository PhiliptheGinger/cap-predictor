import contextlib
import types

import pytest

ql = pytest.importorskip(
    "sentimental_cap_predictor.llm_core.llm_providers.qwen_local",
    reason="qwen provider not available",
)
ds = pytest.importorskip(
    "sentimental_cap_predictor.llm_core.llm_providers.deepseek",
    reason="deepseek provider not available",
)
from sentimental_cap_predictor.llm_core.provider_config import (
    DeepSeekConfig,
    QwenLocalConfig,
)


def _patch_qwen(monkeypatch):
    class DummyTokenizer:
        pass

    class DummyConfig:
        pass

    class DummyModel:
        def tie_weights(self):
            pass

        def eval(self):
            return self

    monkeypatch.setattr(
        ql, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer())
    )
    monkeypatch.setattr(
        ql, "AutoConfig", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyConfig())
    )
    monkeypatch.setattr(
        ql,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_config=lambda *a, **k: DummyModel()),
    )
    monkeypatch.setattr(ql, "init_empty_weights", lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        ql,
        "load_checkpoint_and_dispatch",
        lambda model, checkpoint, device_map=None, offload_folder=None, dtype=None: model,
    )
    monkeypatch.setattr(
        ql,
        "_auto_runtime_prefs",
        lambda: {
            "model_path": "dummy",
            "device_map": "cpu",
            "dtype": None,
            "offload_folder": "/tmp",
        },
    )


@pytest.mark.skipif(not hasattr(ql, "QwenLocalProvider"), reason="provider missing")
def test_qwen_local_provider_init(monkeypatch):
    _patch_qwen(monkeypatch)
    cfg = QwenLocalConfig()
    provider = ql.QwenLocalProvider(**cfg.model_dump())
    assert provider.max_new_tokens == cfg.max_new_tokens


def test_qwen_local_provider_unknown_kwargs():
    with pytest.raises(TypeError):
        ql.QwenLocalProvider(temperature=0.1, max_new_tokens=1, foo=1)


def _patch_deepseek(monkeypatch):
    class DummyResp:
        choices = [types.SimpleNamespace(message={"content": ""})]

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    return DummyResp()

    monkeypatch.setattr(ds, "OpenAI", lambda *a, **k: DummyClient())


@pytest.mark.skipif(not hasattr(ds, "DeepSeekProvider"), reason="provider missing")
def test_deepseek_provider_init(monkeypatch):
    _patch_deepseek(monkeypatch)
    cfg = DeepSeekConfig(api_key="x")
    provider = ds.DeepSeekProvider(**cfg.model_dump())
    assert provider.max_new_tokens == cfg.max_new_tokens


def test_deepseek_provider_unknown_kwargs():
    with pytest.raises(TypeError):
        ds.DeepSeekProvider(temperature=0.1, max_new_tokens=1, foo=1)
