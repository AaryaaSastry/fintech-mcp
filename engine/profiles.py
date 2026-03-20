"""Reusable profile and client overview helpers."""
from __future__ import annotations

from .data_loader import load_dataset


def get_client_overview(client_id: int) -> dict:
    """Return a normalized client overview from transaction history."""
    df, invalid_records = load_dataset()
    df = df[df["client_id"] == client_id].copy()
    if df.empty:
        return {"error": f"No data for client {client_id}"}

    row = df.iloc[0]
    total_spent = float(df["amount"].sum())
    transaction_count = int(len(df))
    yearly_income = float(row.get("yearly_income", 0) or 0)
    total_debt = float(row.get("total_debt", 0) or 0)
    credit_limit = float(row.get("credit_limit", 0) or 0)
    credit_score = int(row.get("credit_score", 0) or 0)
    current_age = int(row.get("current_age", 0) or 0)

    debt_to_income_ratio = total_debt / yearly_income if yearly_income else 0.0
    utilization_percent = (total_spent / credit_limit) * 100 if credit_limit else 0.0
    spending_to_income_ratio = total_spent / yearly_income if yearly_income else 0.0
    avg_transaction = total_spent / transaction_count if transaction_count else 0.0

    return {
        "client_id": client_id,
        "current_age": current_age,
        "yearly_income": yearly_income,
        "total_debt": total_debt,
        "credit_score": credit_score,
        "credit_limit": credit_limit,
        "total_transactions": transaction_count,
        "total_spent": round(total_spent, 2),
        "average_transaction": round(avg_transaction, 2),
        "debt_to_income_ratio": round(debt_to_income_ratio, 3),
        "utilization_percent": round(utilization_percent, 2),
        "spending_to_income_ratio": round(spending_to_income_ratio, 3),
        "debt_risk_level": "high" if debt_to_income_ratio > 0.5 else "medium" if debt_to_income_ratio > 0.3 else "low",
        "utilization_status": "over_limit" if utilization_percent > 100 else "high" if utilization_percent > 75 else "normal",
        "credit_grade": "excellent" if credit_score >= 750 else "good" if credit_score >= 700 else "fair" if credit_score >= 650 else "poor",
        "invalid_records": invalid_records,
    }
