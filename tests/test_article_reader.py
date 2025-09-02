from importlib import util
from pathlib import Path

# Load module directly to avoid importing heavy package dependencies
module_path = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "sentimental_cap_predictor"
    / "news"
    / "article_reader.py"
)
spec = util.spec_from_file_location("article_reader", module_path)
assert spec and spec.loader
article_reader = util.module_from_spec(spec)
spec.loader.exec_module(article_reader)

strip_ads = article_reader.strip_ads
chunk = article_reader.chunk


def test_strip_ads_removes_advertisement_lines():
    text = "Hello\nAdvertisement\nWorld\nSponsored Content\nBye"
    cleaned = strip_ads(text)
    assert "Advertisement" not in cleaned
    assert "Sponsored Content" not in cleaned
    assert "Hello" in cleaned and "World" in cleaned and "Bye" in cleaned


def test_chunk_respects_max_tokens_and_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk(text, max_tokens=4, overlap=1)
    assert chunks == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]
    assert all(len(c.split()) <= 4 for c in chunks)
