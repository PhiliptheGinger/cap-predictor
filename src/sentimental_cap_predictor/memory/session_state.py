from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Mutable state for maintaining context during chatbot sessions."""

    last_article: dict | None = None
    recent_chunks: list[str] = field(default_factory=list)
    last_query: str | None = None

    def set_article(self, article: dict) -> None:
        """Set the most recently referenced article."""
        self.last_article = article

    def clear_article(self) -> None:
        """Clear stored article and related context."""
        self.last_article = None
        self.recent_chunks.clear()
        self.last_query = None


STATE = SessionState()
