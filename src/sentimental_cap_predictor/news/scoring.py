"""News article scoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_news(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    recency_weight: float = 0.5,
    length_weight: float = 0.3,
    credibility_weight: float = 0.2,
    decay: float = 1.0,
    ewma_span: int = 5,
) -> pd.DataFrame:
    """Compute weighted scores for news articles with EWMA smoothing.

    Parameters
    ----------
    df:
        DataFrame containing ``timestamp``, ``length`` and ``credibility``
        columns.
    now:
        Reference time for the recency calculation. Defaults to the most recent
        timestamp in ``df``.
    recency_weight, length_weight, credibility_weight:
        Weights applied to the recency, length and credibility components. They
        do not need to sum to ``1`` but typically should.
    decay:
        Exponential decay factor for the recency component measured in days.
    ewma_span:
        Span parameter for exponential weighted moving average smoothing.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with added ``score`` and ``ewma_score`` columns.
    """
    if now is None:
        now = pd.to_datetime(df["timestamp"]).max()

    df = df.copy()

    timestamp = pd.to_datetime(df["timestamp"])
    age_days = (now - timestamp).dt.total_seconds() / 86400
    recency = np.exp(-decay * age_days)

    length_max = float(df["length"].max())
    if length_max == 0:
        length_max = 1.0
    length_norm = df["length"] / length_max

    score = (
        recency_weight * recency
        + length_weight * length_norm
        + credibility_weight * df["credibility"]
    )
    df["score"] = score
    df["ewma_score"] = score.ewm(span=ewma_span, adjust=False).mean()
    return df


__all__ = ["score_news"]
