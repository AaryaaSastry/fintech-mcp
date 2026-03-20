"""Filtering primitives for analytics queries."""
from __future__ import annotations

import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict | None = None) -> pd.DataFrame:
    """Apply a validated filter dictionary to a dataframe."""
    if not filters:
        return df

    filtered = df
    for column, operations in filters.items():
        for operator, value in operations.items():
            if operator == "eq":
                filtered = filtered[filtered[column] == value]
            elif operator == "in":
                filtered = filtered[filtered[column].isin(value)]
            elif operator == "gt":
                filtered = filtered[filtered[column] > value]
            elif operator == "gte":
                filtered = filtered[filtered[column] >= value]
            elif operator == "lt":
                filtered = filtered[filtered[column] < value]
            elif operator == "lte":
                filtered = filtered[filtered[column] <= value]
            elif operator == "between":
                lower, upper = value
                filtered = filtered[filtered[column].between(lower, upper)]
            else:
                raise ValueError(f"Unsupported filter operator '{operator}'")

    return filtered


def apply_recent_window(df: pd.DataFrame, recent_months: int | None = None) -> pd.DataFrame:
    """Filter records to the most recent N months relative to the max timestamp."""
    if not recent_months or "timestamp" not in df.columns or df.empty:
        return df

    latest_timestamp = df["timestamp"].max()
    if pd.isna(latest_timestamp):
        return df

    cutoff = latest_timestamp - pd.Timedelta(days=recent_months * 30)
    return df[df["timestamp"] >= cutoff]
