"""Utility to fetch a URL and extract readable text and metadata."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import requests


def read_url(url: str) -> Dict[str, Any]:
    """Fetch content from ``url`` and extract text with metadata.

    The function performs a basic content type check. HTML pages are parsed
    with :mod:`trafilatura` when available, otherwise a very small fallback
    based on :mod:`lxml` is used. PDF documents are handled via :mod:`pypdf`.

    Parameters
    ----------
    url:
        The address to fetch.

    Returns
    -------
    dict
        Dictionary with ``text`` and ``meta`` keys.
    """

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    meta: Dict[str, Any] = {"content_type": content_type, "url": response.url}

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError("pypdf is required for PDF extraction") from exc

        reader = PdfReader(BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        doc_meta = reader.metadata or {}
        meta.update({k[1:]: v for k, v in doc_meta.items() if k and v})
        meta["pages"] = len(reader.pages)
        return {"text": text, "meta": meta}

    # HTML or plain text
    html = response.text
    text = ""
    try:
        import trafilatura  # type: ignore

        text = trafilatura.extract(html) or ""
        try:  # metadata extraction is optional
            from trafilatura.metadata import extract_metadata  # type: ignore

            md = extract_metadata(html)
            if md:
                meta.update(
                    {
                        k: v
                        for k, v in md.__dict__.items()
                        if not k.startswith("_") and v
                    }
                )
        except Exception:
            pass
    except Exception:
        pass

    if not text:
        try:
            from lxml import html as lh  # type: ignore

            root = lh.fromstring(html)
            text = " ".join(root.itertext())
            title = root.findtext(".//title")
            if title:
                meta.setdefault("title", title.strip())
        except Exception:
            text = html

    if "title" not in meta:
        try:
            from lxml import html as lh  # type: ignore

            title = lh.fromstring(html).findtext(".//title")
            if title:
                meta["title"] = title.strip()
        except Exception:
            pass

    return {"text": text, "meta": meta}


__all__ = ["read_url"]


# Optional agent tool registration
try:  # pragma: no cover - registration is optional at runtime
    from pydantic import BaseModel

    from sentimental_cap_predictor.llm_core.agent.tool_registry import (
        ToolSpec,
        register_tool,
    )

    class ReadUrlInput(BaseModel):
        url: str

    class ReadUrlOutput(BaseModel):
        text: str
        meta: Dict[str, Any]

    def _read_url_handler(payload: ReadUrlInput) -> ReadUrlOutput:
        result = read_url(payload.url)
        return ReadUrlOutput(**result)

    register_tool(
        ToolSpec(
            name="read.url",
            input_model=ReadUrlInput,
            output_model=ReadUrlOutput,
            handler=_read_url_handler,
        )
    )
except Exception:  # pragma: no cover - silently ignore registration issues
    pass
