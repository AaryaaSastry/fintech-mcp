"""Central reusable analytics service."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loader import load_dataset
from .filters import apply_filters, apply_recent_window
from .grouping import apply_time_grain, get_group_fields, TIME_GRAIN_COLUMN
from .metrics import compute_metrics
from .validators import validate_request


def _replace_internal_keys(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        item = dict(record)
        if TIME_GRAIN_COLUMN in item:
            item["time_grain"] = item.pop(TIME_GRAIN_COLUMN)
        normalized.append(item)
    return normalized


def run_analysis(
    filters: dict = None,
    group_by: list = None,
    metrics: list = None,
    sort_by: str = None,
    order: str = "desc",
    limit: int = 100,
    time_grain: str = None,
    recent_months: int | None = None,
):
    """Run a parameter-driven analytics query against the dataset."""
    validate_request(filters, group_by, metrics, sort_by, order, limit, time_grain, recent_months)
    normalized_sort_by = TIME_GRAIN_COLUMN if sort_by == "time_grain" else sort_by

    df, invalid_records = load_dataset()
    filtered = apply_filters(df, filters)
    latest_timestamp = filtered["timestamp"].max() if "timestamp" in filtered.columns and not filtered.empty else None
    cutoff_timestamp = None
    if recent_months is not None and latest_timestamp is not None:
        cutoff_timestamp = latest_timestamp - pd.Timedelta(days=recent_months * 30)
    filtered = apply_recent_window(filtered, recent_months)
    filtered = apply_time_grain(filtered, time_grain)
    group_fields = get_group_fields(group_by, time_grain)

    result = compute_metrics(filtered, group_fields, metrics)

    if normalized_sort_by:
        ascending = order == "asc"
        result = result.sort_values(by=normalized_sort_by, ascending=ascending, kind="stable")

    result = result.head(limit)
    records = _replace_internal_keys(result.to_dict(orient="records"))

    return {
        "data": records,
        "meta": {
            "row_count": len(records),
            "filters_applied": filters or {},
            "invalid_records": invalid_records,
            "recent_months": recent_months,
            "recent_window_anchor": latest_timestamp.strftime("%Y-%m-%d") if latest_timestamp is not None else None,
            "recent_window_cutoff": cutoff_timestamp.strftime("%Y-%m-%d") if cutoff_timestamp is not None else None,
        },
    }
