import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def _load_frontend():
    sys.modules["faiss"] = types.ModuleType("faiss")
    sys.modules["torch"] = types.ModuleType("torch")
    dummy_trans = types.ModuleType("transformers")
    class AutoModel:
        pass
    class AutoTokenizer:
        pass
    dummy_trans.AutoModel = AutoModel
    dummy_trans.AutoTokenizer = AutoTokenizer
    sys.modules["transformers"] = dummy_trans
    dummy_dotenv = types.ModuleType("dotenv")
    dummy_dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dummy_dotenv

    dummy_news = types.ModuleType("sentimental_cap_predictor.data.news")
    @dataclass
    class FetchArticleSpec:
        query: str = ""
    dummy_news.FetchArticleSpec = FetchArticleSpec
    dummy_news.fetch_first_gdelt_article = lambda *a, **k: None
    dummy_news.fetch_article = lambda *a, **k: None
    dummy_data = types.ModuleType("sentimental_cap_predictor.data")
    dummy_data.__path__ = []
    dummy_data.news = dummy_news
    sys.modules["sentimental_cap_predictor.data"] = dummy_data
    sys.modules["sentimental_cap_predictor.data.news"] = dummy_news

    spec = importlib.util.spec_from_file_location(
        "chatbot_frontend",
        Path(__file__).resolve().parents[1]
        / "src"
        / "sentimental_cap_predictor"
        / "llm_core"
        / "chatbot_frontend.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_disabled_by_default(monkeypatch):
    cf = _load_frontend()
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = cf._parse_args()
    assert args.agent is False


def test_agent_enabled_via_flag(monkeypatch):
    cf = _load_frontend()
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--agent"])
    args = cf._parse_args()
    assert args.agent is True


def test_agent_enabled_via_env(monkeypatch):
    cf = _load_frontend()
    monkeypatch.setenv("AGENT_MODE", "1")
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = cf._parse_args()
    assert args.agent is True
