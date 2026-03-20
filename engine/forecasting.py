"""Reusable forecast helpers."""
from __future__ import annotations

import pandas as pd

from .data_loader import load_dataset


def _build_confidence(monthly: pd.Series, latest_timestamp: pd.Timestamp | None) -> tuple[str, list[str]]:
    """Classify forecast confidence from history depth and recency."""
    warnings: list[str] = []
    months_count = int(len(monthly))

    if months_count < 6:
        warnings.append("limited_history")

    if months_count > 1:
        month_index = monthly.index.to_timestamp()
        span_days = int((month_index.max() - month_index.min()).days)
        if span_days > months_count * 45:
            warnings.append("sparse_month_coverage")

    if latest_timestamp is not None and not pd.isna(latest_timestamp):
        stale_days = int((pd.Timestamp.now(tz="UTC").tz_localize(None) - latest_timestamp).days)
        if stale_days > 365:
            warnings.append("stale_history")

    if "limited_history" in warnings or "stale_history" in warnings:
        return "low", warnings
    if "sparse_month_coverage" in warnings:
        return "medium", warnings
    return "high", warnings


def run_forecast_analysis(client_id: int | None = None, months: int = 3) -> dict:
    """Average-based spending projection for a client or all users."""
    if months <= 0:
        raise ValueError("months must be greater than 0")

    df, invalid_records = load_dataset()
    if client_id is not None:
        df = df[df["client_id"] == client_id].copy()

    if df.empty:
        target = f"client {client_id}" if client_id is not None else "all users"
        return {"error": f"No data available for {target}"}

    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum().sort_index()
    avg_monthly = float(monthly.mean()) if not monthly.empty else 0.0
    latest_timestamp = df["timestamp"].max() if "timestamp" in df.columns else None
    earliest_timestamp = df["timestamp"].min() if "timestamp" in df.columns else None

    average_delta = float(monthly.diff().dropna().mean()) if len(monthly) > 1 else 0.0
    if average_delta > 0:
        trend = "increasing"
    elif average_delta < 0:
        trend = "decreasing"
    else:
        trend = "stable"

    confidence, warnings = _build_confidence(monthly, latest_timestamp)
    forecast = {f"month_{index + 1}": round(avg_monthly, 2) for index in range(months)}
    return {
        "client_id": client_id,
        "forecast": forecast,
        "avg_historical_monthly": round(avg_monthly, 2),
        "trend": trend,
        "total_months_data": int(len(monthly)),
        "history_start_month": str(monthly.index.min()) if not monthly.empty else None,
        "history_end_month": str(monthly.index.max()) if not monthly.empty else None,
        "latest_transaction_date": latest_timestamp.strftime("%Y-%m-%d") if latest_timestamp is not None and not pd.isna(latest_timestamp) else None,
        "data_span_days": int((latest_timestamp - earliest_timestamp).days) if latest_timestamp is not None and earliest_timestamp is not None and not pd.isna(latest_timestamp) and not pd.isna(earliest_timestamp) else 0,
        "confidence": confidence,
        "warnings": warnings,
        "invalid_records": invalid_records,
    }
