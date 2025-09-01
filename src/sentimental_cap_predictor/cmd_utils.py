"""Command parsing helpers for chatbot frontend."""

from __future__ import annotations


def extract_cmd(text: str) -> tuple[str | None, str | None]:
    """Return either a command or a single question from ``text``.

    The function looks for a ``CMD:`` prefix to indicate a shell command. If
    the response is a lone question, ending with ``?`` and containing no
    newlines, it is returned as such. When neither pattern matches ``(None,
    None)`` is returned.
    """

    import re

    text = text.strip()
    if match := re.fullmatch(r"CMD:\s*(.+)", text, re.DOTALL):
        return match.group(1).strip(), None
    if text.endswith("?") and text.count("?") == 1 and "\n" not in text:
        return None, text
    return None, None
