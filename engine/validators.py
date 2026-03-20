"""Validation helpers for modular analytics requests."""
from __future__ import annotations

from typing import Any

from .grouping import TIME_GRAIN_COLUMN
from .metrics import METRIC_REGISTRY, parse_metric


ALLOWED_COLUMNS = {
    "transaction_id",
    "client_id",
    "card_id",
    "card_type",
    "card_brand",
    "use_chip",
    "mcc_code",
    "merchant_id",
    "merchant_category",
    "merchant_city",
    "amount",
    "is_fraud",
    "timestamp",
    "current_age",
    "total_debt",
    "credit_limit",
    "credit_score",
    "yearly_income",
}

ALLOWED_OPERATORS = {"eq", "in", "gt", "gte", "lt", "lte", "between"}
ALLOWED_TIME_GRAINS = {"day", "week", "month"}
MAX_LIMIT = 1000


def validate_filters(filters: dict | None) -> None:
    if not filters:
        return

    for column, operations in filters.items():
        if column not in ALLOWED_COLUMNS:
            raise ValueError(f"Invalid filter column '{column}'")
        if not isinstance(operations, dict) or not operations:
            raise ValueError(f"Filter for column '{column}' must be a non-empty object")

        for operator, value in operations.items():
            if operator not in ALLOWED_OPERATORS:
                raise ValueError(f"Invalid operator '{operator}' for column '{column}'")
            if operator == "in" and not isinstance(value, (list, tuple, set)):
                raise ValueError(f"Operator 'in' for column '{column}' must use a list-like value")
            if operator == "between":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"Operator 'between' for column '{column}' must use exactly two values")


def validate_group_by(group_by: list | None) -> None:
    if not group_by:
        return

    for column in group_by:
        if column not in ALLOWED_COLUMNS:
            raise ValueError(f"Invalid group_by column '{column}'")


def validate_metrics(metrics: list | None) -> None:
    if not metrics:
        raise ValueError("At least one metric must be provided")

    for metric in metrics:
        metric_name, metric_arg = parse_metric(metric)
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Unsupported metric '{metric_name}'")

        definition = METRIC_REGISTRY[metric_name]
        if definition.requires_argument:
            if metric_arg is None:
                raise ValueError(f"Metric '{metric_name}' requires an argument")
            if metric_name == "count" and metric_arg != "*":
                raise ValueError("count metric must use count(*)")
            if metric_name != "count" and metric_arg not in ALLOWED_COLUMNS:
                raise ValueError(f"Metric '{metric}' uses invalid column '{metric_arg}'")
        elif metric_arg is not None:
            raise ValueError(f"Metric '{metric_name}' does not take an argument")


def validate_sort_by(sort_by: str | None, group_by: list | None, metrics: list | None, time_grain: str | None) -> None:
    if sort_by is None:
        return

    allowed_sort_fields = set(group_by or [])
    allowed_sort_fields.update(metrics or [])
    if time_grain:
        allowed_sort_fields.add("time_grain")
        allowed_sort_fields.add(TIME_GRAIN_COLUMN)

    if sort_by not in allowed_sort_fields:
        raise ValueError(f"sort_by '{sort_by}' must match a group_by field or metric")


def validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if limit > MAX_LIMIT:
        raise ValueError(f"limit must be <= {MAX_LIMIT}")


def validate_recent_months(recent_months: int | None) -> None:
    if recent_months is not None and recent_months <= 0:
        raise ValueError("recent_months must be greater than 0")


def validate_order(order: str) -> None:
    if order not in {"asc", "desc"}:
        raise ValueError("order must be either 'asc' or 'desc'")


def validate_time_grain(time_grain: str | None) -> None:
    if time_grain and time_grain not in ALLOWED_TIME_GRAINS:
        raise ValueError(f"time_grain must be one of {sorted(ALLOWED_TIME_GRAINS)}")


def validate_request(
    filters: dict | None,
    group_by: list | None,
    metrics: list | None,
    sort_by: str | None,
    order: str,
    limit: int,
    time_grain: str | None,
    recent_months: int | None = None,
) -> None:
    validate_filters(filters)
    validate_group_by(group_by)
    validate_metrics(metrics)
    validate_time_grain(time_grain)
    validate_sort_by(sort_by, group_by, metrics, time_grain)
    validate_order(order)
    validate_limit(limit)
    validate_recent_months(recent_months)
