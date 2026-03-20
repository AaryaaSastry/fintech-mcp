"""Metric registry and aggregation helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd


MetricFunc = Callable[[pd.DataFrame, list[str], str | None], pd.Series | float | int]
_METRIC_PATTERN = re.compile(r"^(?P<name>[a-z_]+)(?:\((?P<arg>[\w\*]+)\))?$")


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    requires_argument: bool
    func: MetricFunc


METRIC_REGISTRY: dict[str, MetricDefinition] = {}


def register_metric(name: str, requires_argument: bool):
    """Register a metric definition by name."""

    def decorator(func: MetricFunc):
        METRIC_REGISTRY[name] = MetricDefinition(
            name=name,
            requires_argument=requires_argument,
            func=func,
        )
        return func

    return decorator


def parse_metric(metric: str) -> tuple[str, str | None]:
    """Parse a metric expression like `sum(amount)`."""
    match = _METRIC_PATTERN.fullmatch(metric.strip())
    if not match:
        raise ValueError(f"Invalid metric expression '{metric}'")
    return match.group("name"), match.group("arg")


def list_supported_metrics() -> list[str]:
    return sorted(METRIC_REGISTRY.keys())


def compute_metrics(df: pd.DataFrame, group_fields: list[str], metrics: list[str]) -> pd.DataFrame:
    """Compute validated metrics against the dataframe."""
    if not metrics:
        raise ValueError("At least one metric must be provided")

    grouped = df.groupby(group_fields, dropna=False) if group_fields else None
    result = pd.DataFrame(index=grouped.size().index if grouped is not None else [0])

    for metric in metrics:
        metric_name, metric_arg = parse_metric(metric)
        definition = METRIC_REGISTRY[metric_name]
        result[metric] = definition.func(df, group_fields, metric_arg)

    if group_fields:
        return result.reset_index()
    return result.reset_index(drop=True)


@register_metric("sum", requires_argument=True)
def metric_sum(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if group_fields:
        return df.groupby(group_fields, dropna=False)[column].sum()
    return float(df[column].sum())


@register_metric("avg", requires_argument=True)
def metric_avg(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if group_fields:
        return df.groupby(group_fields, dropna=False)[column].mean()
    return float(df[column].mean())


@register_metric("count", requires_argument=True)
def metric_count(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if column != "*":
        raise ValueError("count metric must use count(*)")
    if group_fields:
        return df.groupby(group_fields, dropna=False).size()
    return int(len(df))


@register_metric("min", requires_argument=True)
def metric_min(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if group_fields:
        return df.groupby(group_fields, dropna=False)[column].min()
    return float(df[column].min())


@register_metric("max", requires_argument=True)
def metric_max(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if group_fields:
        return df.groupby(group_fields, dropna=False)[column].max()
    return float(df[column].max())


@register_metric("fraud_rate", requires_argument=False)
def metric_fraud_rate(df: pd.DataFrame, group_fields: list[str], column: str | None):
    fraud_series = pd.to_numeric(df["is_fraud"], errors="coerce")
    valid_fraud = fraud_series.where(fraud_series.isin([0, 1]))
    if group_fields:
        temp = df.copy()
        temp["__valid_is_fraud"] = valid_fraud
        return temp.groupby(group_fields, dropna=False)["__valid_is_fraud"].mean()
    return float(valid_fraud.mean())


@register_metric("avg_ticket", requires_argument=False)
def metric_avg_ticket(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if group_fields:
        grouped = df.groupby(group_fields, dropna=False)["amount"]
        return grouped.sum() / grouped.size()
    if len(df) == 0:
        return 0.0
    return float(df["amount"].sum() / len(df))


@register_metric("distinct_count", requires_argument=True)
def metric_distinct_count(df: pd.DataFrame, group_fields: list[str], column: str | None):
    if group_fields:
        return df.groupby(group_fields, dropna=False)[column].nunique()
    return int(df[column].nunique())
