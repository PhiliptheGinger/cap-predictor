from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class DataBundle:
    """Container for point-in-time market and sentiment data.

    All data frames must share the same :class:`~pandas.DatetimeIndex` and be
    aligned so that strategies cannot accidentally access future information.
    """

    prices: pd.DataFrame
    features: Optional[pd.DataFrame] = None
    sentiment: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, object]] = None

    def validate(self) -> "DataBundle":
        """Validate that all included data are aligned by timestamp.

        Returns the bundle itself so callers can use ``bundle.validate()`` when
        constructing the object.  The check is intentionally lightweight – it
        ensures that all frames share the same, monotonically increasing
        ``DatetimeIndex`` and that no timestamps lie in the future.
        """

        base_index = self.prices.index
        if not isinstance(base_index, pd.DatetimeIndex):
            raise ValueError("prices must be indexed by pandas.DatetimeIndex")
        if not base_index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted chronologically")
        if (base_index > pd.Timestamp.utcnow()).any():
            raise ValueError("prices contain timestamps in the future")

        for frame in (self.features, self.sentiment):
            if frame is None:
                continue
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError("all DataFrames must use pandas.DatetimeIndex")
            if not frame.index.equals(base_index):
                raise ValueError("DataFrames must be aligned on the same index")
            if (frame.index > pd.Timestamp.utcnow()).any():
                raise ValueError("data contains timestamps in the future")
        return self

