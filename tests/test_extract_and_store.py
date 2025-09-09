import importlib.util
import sys
from pathlib import Path
import types

# Set up lightweight stubs for extraction dependencies

class DummyConfig(dict):
    def set(self, section, option, value):
        self[(section, option)] = value

def dummy_use_config():
    return DummyConfig()

trafilatura_stub = types.SimpleNamespace(
    extract=lambda html, url=None, include_comments=False, include_tables=False, config=None: "extracted text",
)
settings_stub = types.SimpleNamespace(use_config=dummy_use_config)

sys.modules.setdefault("trafilatura", trafilatura_stub)
sys.modules.setdefault("trafilatura.settings", settings_stub)
# Provide a minimal requests module for fetch_html
requests_stub = types.SimpleNamespace(get=lambda *a, **k: None)
sys.modules.setdefault("requests", requests_stub)

root = Path(__file__).resolve().parents[1]

extract_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.news.extract",
    root / "src" / "sentimental_cap_predictor" / "news" / "extract.py",
)
extract = importlib.util.module_from_spec(extract_spec)
sys.modules["sentimental_cap_predictor.news.extract"] = extract
extract_spec.loader.exec_module(extract)

vector_spec = importlib.util.spec_from_file_location(
    "sentimental_cap_predictor.memory.vector_store",
    root / "src" / "sentimental_cap_predictor" / "memory" / "vector_store.py",
)
vector_store = importlib.util.module_from_spec(vector_spec)
sys.modules["sentimental_cap_predictor.memory.vector_store"] = vector_store
vector_spec.loader.exec_module(vector_store)


def test_extract_main_text(monkeypatch):
    class DummyResp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    html = "<html><body><p>Hello world</p></body></html>"

    def fake_get(url, headers=None, timeout=0):  # noqa: ANN001
        return DummyResp(html)

    monkeypatch.setattr(extract.requests, "get", fake_get)
    fetched = extract.fetch_html("http://example.com")
    text = extract.extract_main_text(fetched, url="http://example.com")
    assert text


def test_upsert_and_query(tmp_path, monkeypatch):
    class DummyCollection:
        def __init__(self):
            self.store = {}

        def upsert(self, ids=None, embeddings=None, metadatas=None):  # noqa: ANN001
            self.store[ids[0]] = metadatas[0]

        add = upsert

        def query(self, query_embeddings=None, n_results=5, include=None):  # noqa: ANN001
            if not self.store:
                return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
            doc_id, meta = next(iter(self.store.items()))
            return {
                "ids": [[doc_id]],
                "distances": [[0.0]],
                "metadatas": [[meta]],
            }

    class DummyClient:
        def __init__(self, path):
            self.path = path
            self.collection = DummyCollection()

        def get_or_create_collection(self, name):  # noqa: ANN001
            return self.collection

    dummy_chroma = types.SimpleNamespace(PersistentClient=lambda path: DummyClient(path))
    monkeypatch.setitem(sys.modules, "chromadb", dummy_chroma)
    monkeypatch.setattr(vector_store.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    monkeypatch.setattr(vector_store, "_embed", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

    vector_store._INDEX = None
    vector_store._USING_PINECONE = False

    vector_store.upsert("doc1", "sample text", {"source": "test"})
    results = vector_store.query("sample")

    assert results
    assert results[0]["id"] == "doc1"
    assert results[0]["metadata"]["source"] == "test"
