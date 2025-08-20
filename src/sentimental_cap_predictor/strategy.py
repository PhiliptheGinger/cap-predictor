from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .data_bundle import DataBundle


class StrategyIdea(ABC):
    """Abstract base class for model-generated strategies."""

    @abstractmethod
    def generate_signals(self, data: DataBundle) -> pd.Series:
        """Return target position signals indexed by date."""
        raise NotImplementedError


class BuyAndHoldStrategy(StrategyIdea):
    """Simple example strategy used as guidance for LLM-generated code."""

    def generate_signals(self, data: DataBundle) -> pd.Series:  # pragma: no cover - trivial
        return pd.Series(1, index=data.prices.index)
