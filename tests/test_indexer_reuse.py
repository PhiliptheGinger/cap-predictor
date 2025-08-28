import sys
import types
from pathlib import Path

import numpy as np

# Stub torch
class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _Torch(types.SimpleNamespace):
    def no_grad(self):
        return _NoGrad()


torch_stub = _Torch()
sys.modules.setdefault("torch", torch_stub)

# Stub faiss
class _Faiss(types.SimpleNamespace):
    class IndexFlatL2:
        def __init__(self, dim):
            self.dim = dim
            self.vectors = None

        def add(self, embeddings):
            self.vectors = np.array(embeddings)

    @staticmethod
    def write_index(index, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, index.vectors)


faiss_stub = _Faiss()
sys.modules.setdefault("faiss", faiss_stub)

# Stub transformers
class _Transformers(types.SimpleNamespace):
    class AutoModel:
        pass

    class AutoTokenizer:
        pass


sys.modules.setdefault("transformers", _Transformers())

# Stub loguru logger
loguru_stub = types.SimpleNamespace(
    logger=types.SimpleNamespace(info=lambda *a, **k: None)
)
sys.modules.setdefault("loguru", loguru_stub)

# Import indexer module directly to avoid package side effects
import importlib.util

indexer_path = Path(__file__).resolve().parents[1] / "src" / "sentimental_cap_predictor" / "indexer.py"
spec = importlib.util.spec_from_file_location("indexer", indexer_path)
indexer = importlib.util.module_from_spec(spec)
sys.modules["sentimental_cap_predictor.indexer"] = indexer
spec.loader.exec_module(indexer)


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        arr = np.array([[len(t)] for t in texts], dtype=np.float32)
        return {"input_ids": arr}


class Tensor:
    def __init__(self, arr):
        self.arr = np.array(arr, dtype=np.float32)

    def mean(self, dim, keepdim=False):
        return Tensor(self.arr.mean(axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def astype(self, dtype):
        return self.arr.astype(dtype)


class DummyModel:
    def __call__(self, **inputs):
        arr = inputs["input_ids"][:, :, None]
        return types.SimpleNamespace(last_hidden_state=Tensor(arr))


def dummy_load_model(model_name=indexer.MODEL_NAME):
    return DummyTokenizer(), DummyModel()


def test_embed_texts_reuse_model(monkeypatch):
    monkeypatch.setattr(indexer, "_load_model", dummy_load_model)
    texts = ["foo", "bar"]
    baseline = indexer.embed_texts(texts)
    tok, mod = dummy_load_model()
    reused = indexer.embed_texts(texts, tokenizer=tok, model=mod)
    assert np.allclose(baseline, reused)


def test_build_index_reuse_model(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer, "_load_model", dummy_load_model)
    papers = [
        {"title": "A", "abstract": "B"},
        {"title": "C", "abstract": "D"},
    ]
    path1 = tmp_path / "a.npy"
    path2 = tmp_path / "b.npy"
    indexer.build_index(papers, path1)
    tok, mod = dummy_load_model()
    indexer.build_index(papers, path2, tokenizer=tok, model=mod)
    emb1 = np.load(path1)
    emb2 = np.load(path2)
    assert np.allclose(emb1, emb2)
