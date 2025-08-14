"""Helper utilities for preparing data before modeling.

This module centralizes common preprocessing steps such as
scaling price data, cleaning missing values and merging new
results with existing datasets.  The goal is to keep CLI
scripts small by reusing these helpers.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from .modeling.preprocessing import (
    clean_data,
    handle_missing_values,
    feature_engineering,
)


def preprocess_price_data(price_df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Clean and scale raw price data.

    Parameters
    ----------
    price_df:
        Raw price dataframe containing at least a ``close`` column.

    Returns
    -------
    Tuple[pd.DataFrame, MinMaxScaler]
        The processed dataframe and the fitted scaler used on the
        ``close`` column.
    """
    scaler = MinMaxScaler()
    price_df = price_df.copy()
    price_df["close"] = scaler.fit_transform(price_df[["close"]])

    # Reuse existing helpers for further cleaning
    price_df = clean_data(price_df)
    price_df = handle_missing_values(price_df)
    price_df = feature_engineering(price_df)

    return price_df, scaler


def merge_data(existing_df: pd.DataFrame, new_df: pd.DataFrame, merge_on: str = "Date") -> pd.DataFrame:
    """Merge new data into an existing dataframe avoiding duplicates.

    Parameters
    ----------
    existing_df, new_df:
        Dataframes to merge.
    merge_on:
        Column name used to identify duplicates.
    """
    if merge_on in existing_df.columns:
        existing_df[merge_on] = pd.to_datetime(existing_df[merge_on], errors="coerce")
    if merge_on in new_df.columns:
        new_df[merge_on] = pd.to_datetime(new_df[merge_on], errors="coerce")

    if not existing_df.empty:
        merged_df = (
            pd.concat([existing_df, new_df])
            .drop_duplicates(subset=[merge_on])
            .sort_values(by=merge_on)
            .reset_index(drop=True)
        )
    else:
        merged_df = new_df

    return merged_df
