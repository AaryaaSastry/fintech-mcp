"""Thin tool wrappers that delegate to the shared analytics engine."""
from __future__ import annotations

from engine.analysis_service import run_analysis


def spending_by_city(client_id: int):
    return run_analysis(
        filters={"client_id": {"eq": client_id}},
        group_by=["merchant_city"],
        metrics=["sum(amount)"],
        sort_by="sum(amount)",
        limit=10,
    )


def fraud_by_card_type():
    return run_analysis(
        group_by=["card_type"],
        metrics=["count(*)", "fraud_rate"],
        sort_by="fraud_rate",
        limit=10,
    )


def monthly_spending(client_id: int):
    return run_analysis(
        filters={"client_id": {"eq": client_id}},
        metrics=["sum(amount)", "count(*)"],
        time_grain="month",
        sort_by="time_grain",
        order="asc",
        limit=24,
    )
