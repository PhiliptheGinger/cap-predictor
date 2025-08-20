from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class DataBundle:
    """Container for point-in-time market and sentiment data.

    All data frames must share the same :class:`~pandas.DatetimeIndex` and be
    aligned so that strategies cannot accidentally access future information.

    For multi-asset bundles each data frame is expected to use a two-level
    :class:`~pandas.MultiIndex` on the columns where the first level contains
    the asset identifier and the second level contains the field name, e.g.
    ``('AAPL', 'close')``.  Single-asset bundles may use a flat column index.
    """

    prices: pd.DataFrame
    features: Optional[pd.DataFrame] = None
    sentiment: Optional[pd.DataFrame] = None
    publication_times: Optional[Dict[str, pd.DataFrame]] = None
    metadata: Optional[Dict[str, object]] = None

    # ------------------------------------------------------------------
    # Convenience data accessors
    # ------------------------------------------------------------------
    def _get_from_frame(self, frame: pd.DataFrame, asset: str, field: str) -> pd.Series:
        """Extract a series from ``frame`` for ``asset`` and ``field``.

        Parameters
        ----------
        frame:
            The data frame to query.
        asset:
            Asset identifier such as a ticker symbol.
        field:
            Column/field name to retrieve for the asset.
        """

        if isinstance(frame.columns, pd.MultiIndex):
            try:
                return frame[(asset, field)]
            except KeyError as exc:  # pragma: no cover - defensive
                raise KeyError(f"{asset}/{field} not found") from exc

        # Single asset; rely on metadata to ensure the requested asset matches
        if self.metadata and self.metadata.get("ticker") not in (None, asset):
            raise KeyError(f"asset {asset} not available")
        if field not in frame.columns:
            raise KeyError(f"{field} not found for {asset}")
        return frame[field]

    def get_series(self, asset: str, field: str) -> pd.Series:
        """Return a price series for ``asset`` and ``field``."""

        return self._get_from_frame(self.prices, asset, field)

    def get_feature(self, asset: str, field: str) -> pd.Series:
        """Return a feature series for ``asset`` and ``field``."""

        if self.features is None:
            raise KeyError("no features available")
        return self._get_from_frame(self.features, asset, field)

    def get_sentiment(self, asset: str, field: str) -> pd.Series:
        """Return a sentiment/alternative data series."""

        if self.sentiment is None:
            raise KeyError("no sentiment data available")
        return self._get_from_frame(self.sentiment, asset, field)

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
        now = pd.Timestamp.utcnow().tz_localize(None)
        if (base_index > now).any():
            raise ValueError("prices contain timestamps in the future")

        frames = {
            "prices": self.prices,
            "features": self.features,
            "sentiment": self.sentiment,
        }
        for name, frame in frames.items():
            if frame is None:
                continue
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError("all DataFrames must use pandas.DatetimeIndex")
            if not frame.index.equals(base_index):
                raise ValueError("DataFrames must be aligned on the same index")
            if (frame.index > now).any():
                raise ValueError("data contains timestamps in the future")

            if self.publication_times and name in self.publication_times:
                pub_df = self.publication_times[name]
                if not isinstance(pub_df.index, pd.DatetimeIndex):
                    raise ValueError("publication timestamps must use pandas.DatetimeIndex")
                if not pub_df.index.equals(base_index):
                    raise ValueError("publication timestamps must align with data index")
                if not pub_df.columns.equals(frame.columns):
                    raise ValueError("publication timestamps must align with data columns")
                for col in pub_df.columns:
                    series = pub_df[col]
                    if (series > now).any():
                        raise ValueError("publication timestamps in the future")
                    if (series > base_index).any():
                        raise ValueError("publication timestamp after data timestamp detected")

        return self

