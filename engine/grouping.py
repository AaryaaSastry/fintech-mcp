"""Grouping helpers for reusable analytics queries."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


TIME_GRAIN_COLUMN = "__time_grain"


def apply_time_grain(df: pd.DataFrame, time_grain: str | None = None) -> pd.DataFrame:
    """Add a derived time grain column when requested."""
    if not time_grain:
        return df

    if "timestamp" not in df.columns:
        raise ValueError("time_grain requires a 'timestamp' column in the dataset")

    result = df.copy()
    if time_grain == "day":
        result[TIME_GRAIN_COLUMN] = result["timestamp"].dt.strftime("%Y-%m-%d")
    elif time_grain == "week":
        result[TIME_GRAIN_COLUMN] = result["timestamp"].dt.to_period("W").astype(str)
    elif time_grain == "month":
        result[TIME_GRAIN_COLUMN] = result["timestamp"].dt.to_period("M").astype(str)
    else:
        raise ValueError(f"Unsupported time_grain '{time_grain}'")

    return result


def get_group_fields(group_by: list | None = None, time_grain: str | None = None) -> List[str]:
    """Return the final list of grouping dimensions."""
    fields = list(group_by or [])
    if time_grain:
        fields.append(TIME_GRAIN_COLUMN)
    return fields

