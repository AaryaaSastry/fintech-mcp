"""Shared helpers for MCP tool implementations."""

from functools import lru_cache
import time

import pandas as pd

from engine.data_loader import load_dataset

_aggregate_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 60


@lru_cache(maxsize=1)
def get_transactions() -> pd.DataFrame:
    """Load and preprocess transaction data once per server process."""
    try:
        df, _ = load_dataset()
        if "is_fraud" in df.columns:
            df["is_fraud"] = df["is_fraud"].where(df["is_fraud"].isin([0, 1]), 0).fillna(0).astype(int)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def error_response(message: str) -> dict:
    return {"error": message}


def get_cached(key: str, func):
    now = time.time()
    if key in _aggregate_cache:
        cached_time, cached_result = _aggregate_cache[key]
        if now - cached_time < _CACHE_TTL:
            return cached_result

    result = func()
    _aggregate_cache[key] = (now, result)
    return result
