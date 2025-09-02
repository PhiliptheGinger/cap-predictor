from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..data_bundle import DataBundle


@dataclass
class StrategyParameters:
    """Configuration parameters for a strategy."""

    weight: float = 1.0


class StrategyIdea(ABC):
    """Abstract base class for model-generated strategies."""

    def __init__(self, params: StrategyParameters | None = None) -> None:
        self.params = params or StrategyParameters()

    @abstractmethod
    def generate_signals(self, data: DataBundle) -> pd.DataFrame:
        """Return target asset weights indexed by date."""
        raise NotImplementedError


class BuyAndHoldStrategy(StrategyIdea):
    """Simple example strategy used as guidance for LLM-generated code."""

    def generate_signals(self, data: DataBundle) -> pd.DataFrame:  # pragma: no cover - trivial
        if isinstance(data.prices.columns, pd.MultiIndex):
            assets = data.prices.columns.get_level_values(0).unique().tolist()
        else:
            assets = list(data.prices.columns)
        return pd.DataFrame(
            self.params.weight, index=data.prices.index, columns=assets
        )
