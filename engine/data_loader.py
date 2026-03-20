"""Dataset loading and normalization utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
TIMESTAMP_COLUMN = "timestamp"
FRAUD_COLUMN = "is_fraud"

NUMERIC_COLUMNS = {
    "transaction_id",
    "client_id",
    "merchant_id",
    "mcc_code",
    "amount",
    "is_fraud",
    "credit_limit",
    "total_debt",
    "current_age",
    "credit_score",
    "yearly_income",
}


def resolve_data_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the dataset location from an explicit path, env var, or default."""
    if explicit_path is not None:
        return Path(explicit_path)

    env_path = os.getenv("ANALYTICS_DATA_PATH")
    if env_path:
        return Path(env_path)

    return DEFAULT_DATA_PATH


def load_dataset(data_path: str | Path | None = None) -> Tuple[pd.DataFrame, int]:
    """Load the analytics dataset and normalize key fields.

    Returns a tuple of `(dataframe, invalid_records)` where `invalid_records`
    currently tracks invalid fraud labels that were coerced to null.
    """
    path = resolve_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'")

    df = pd.read_csv(path, on_bad_lines="skip")

    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_records = 0
    if FRAUD_COLUMN in df.columns:
        valid_mask = df[FRAUD_COLUMN].isin([0, 1]) | df[FRAUD_COLUMN].isna()
        invalid_records = int((~valid_mask).sum())
        df.loc[~valid_mask, FRAUD_COLUMN] = pd.NA

    return df, invalid_records
