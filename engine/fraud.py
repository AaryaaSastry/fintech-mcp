"""Reusable fraud analysis helpers."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from .data_loader import load_dataset


ML_MODEL_PATH = "models/fraud/fraud_model.pkl"
ML_SCALER_PATH = "models/fraud/scaler.pkl"


def _load_ml_model():
    try:
        if os.path.exists(ML_MODEL_PATH):
            model = joblib.load(ML_MODEL_PATH)
            try:
                model.n_jobs = 1
            except Exception:
                pass
            return model, joblib.load(ML_SCALER_PATH)
    except Exception:
        pass
    return None, None


def _build_ml_features(row):
    amount = float(row.get("amount", 0))
    income = max(float(row.get("yearly_income", 1) or 1), 1)
    age = float(row.get("current_age", 30) or 30)
    ts = row.get("timestamp", pd.Timestamp.now())
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)

    return {
        "amount": amount,
        "amount_log": np.log1p(amount),
        "credit_score": float(row.get("credit_score", 700) or 700),
        "yearly_income": income,
        "total_debt": float(row.get("total_debt", 0) or 0),
        "debt_to_income": float(row.get("total_debt", 0) or 0) / income,
        "credit_limit": float(row.get("credit_limit", 10000) or 10000),
        "utilization": float(row.get("credit_limit", 10000) or 10000) / income,
        "current_age": age,
        "age_group": 1 if age <= 25 else 2 if age <= 35 else 3 if age <= 45 else 4 if age <= 55 else 5,
        "use_chip": int(bool(row.get("use_chip", False))),
        "card_type_encoded": {"Credit": 1, "Debit": 0, "Prepaid": 2}.get(str(row.get("card_type", "")), 0),
        "card_brand_encoded": {"Visa": 0, "Mastercard": 1, "Discover": 2, "Amex": 3}.get(str(row.get("card_brand", "")), 0),
        "hour": ts.hour if hasattr(ts, "hour") else 12,
        "day_of_week": ts.dayofweek if hasattr(ts, "dayofweek") else 0,
        "is_weekend": 1 if (hasattr(ts, "dayofweek") and ts.dayofweek >= 5) else 0,
        "is_night": 1 if (hasattr(ts, "hour") and (ts.hour >= 22 or ts.hour < 6)) else 0,
        "mcc_code": float(row.get("mcc_code", 5411) or 5411),
        "high_amount": 1 if amount > 1000 else 0,
        "low_credit_score": 1 if float(row.get("credit_score", 700) or 700) < 650 else 0,
        "high_debt_ratio": 1 if (float(row.get("total_debt", 0) or 0) / income) > 0.5 else 0,
    }


def _risk_level(probability: float) -> str:
    if probability > 0.5:
        return "high"
    if probability > 0.2:
        return "medium"
    return "low"


def _heuristic_probability(row) -> float:
    probability = 0.05
    if row.get("amount", 0) > 1000:
        probability += 0.2
    if row.get("credit_score", 700) < 600:
        probability += 0.1
    income = row.get("yearly_income", 1) or 1
    if income > 0 and (row.get("total_debt", 0) or 0) / income > 0.5:
        probability += 0.1
    return min(probability, 1.0)


def check_transaction_fraud(transaction_id: int) -> dict:
    df, _ = load_dataset()
    df = df[df["transaction_id"] == transaction_id]
    if df.empty:
        return {"error": f"Transaction {transaction_id} not found"}

    row = df.iloc[0]
    model, scaler = _load_ml_model()
    if model is not None:
        try:
            features = scaler.transform(pd.DataFrame([_build_ml_features(row)]))
            probability = float(model.predict_proba(features)[0][1])
            is_fraud = bool(model.predict(features)[0])
            return {
                "transaction_id": transaction_id,
                "fraud_probability": round(probability, 4),
                "is_fraud": is_fraud,
                "risk_level": _risk_level(probability),
                "model_used": "Random Forest ML",
            }
        except Exception:
            pass

    probability = _heuristic_probability(row)
    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(probability, 4),
        "risk_level": _risk_level(probability),
        "model_used": "Heuristic",
    }


def get_client_fraud_risk(client_id: int) -> dict:
    df, _ = load_dataset()
    df = df[df["client_id"] == client_id]
    if df.empty:
        return {"error": f"No transactions for client {client_id}"}

    model, scaler = _load_ml_model()
    total_risk = 0.0
    high_value_count = 0
    fraud_transactions = 0

    for _, row in df.iterrows():
        probability = None
        if model is not None:
            try:
                features = scaler.transform(pd.DataFrame([_build_ml_features(row)]))
                probability = float(model.predict_proba(features)[0][1])
                if bool(model.predict(features)[0]):
                    fraud_transactions += 1
            except Exception:
                probability = None

        if probability is None:
            probability = _heuristic_probability(row)
            if probability > 0.5:
                fraud_transactions += 1

        total_risk += probability
        if row.get("amount", 0) > 1000:
            high_value_count += 1

    average_risk = total_risk / len(df)
    return {
        "client_id": client_id,
        "risk_score": round(average_risk, 4),
        "high_value_transactions": high_value_count,
        "total_transactions": int(len(df)),
        "fraud_transactions_detected": int(fraud_transactions),
        "risk_level": _risk_level(average_risk),
        "model_used": "Random Forest ML" if model is not None else "Heuristic",
    }


def get_merchant_fraud_summary(mcc_code: str) -> dict:
    df, _ = load_dataset()
    df = df[df["mcc_code"].astype(str) == str(mcc_code)]
    if df.empty:
        return {"error": f"No data for MCC {mcc_code}"}

    fraud_rate = float(df["is_fraud"].mean()) if "is_fraud" in df.columns else 0.0
    return {
        "mcc_code": str(mcc_code),
        "fraud_rate": round(fraud_rate, 4),
        "total_transactions": int(len(df)),
        "avg_amount": round(float(df["amount"].mean()), 2),
        "risk_level": _risk_level(fraud_rate),
    }


def get_high_risk_transactions(client_id: int, threshold: float = 500.0) -> dict:
    df, _ = load_dataset()
    df = df[(df["client_id"] == client_id) & (df["amount"] > threshold)].copy()
    if df.empty:
        return {"transactions": [], "count": 0, "message": f"No transactions above ${threshold}"}

    return {
        "transactions": df[["transaction_id", "amount", "merchant_category", "timestamp"]].to_dict("records"),
        "count": int(len(df)),
        "threshold": float(threshold),
    }


def get_global_fraud_summary() -> dict:
    df, _ = load_dataset()
    if df.empty or "is_fraud" not in df.columns:
        return {"error": "No fraud data available"}

    client_fraud = (
        df.groupby("client_id")
        .agg(transaction_count=("transaction_id", "count"), fraud_count=("is_fraud", "sum"))
        .reset_index()
    )
    client_fraud["fraud_rate"] = client_fraud["fraud_count"] / client_fraud["transaction_count"]
    top_fraudulent = client_fraud.nlargest(10, "fraud_rate")

    fraud_by_category = df.groupby("merchant_category")["is_fraud"].agg(["sum", "mean"])
    return {
        "total_transactions": int(len(df)),
        "fraud_count": int(df["is_fraud"].sum()),
        "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        "total_clients": int(df["client_id"].nunique()),
        "clients_with_fraud": int((client_fraud["fraud_count"] > 0).sum()),
        "top_fraudulent_clients": [
            {
                "client_id": int(row["client_id"]),
                "fraud_rate": round(float(row["fraud_rate"]), 4),
                "fraud_count": int(row["fraud_count"]),
            }
            for _, row in top_fraudulent.iterrows()
        ],
        "fraud_by_category": {
            category: {"count": int(values["sum"]), "rate": float(values["mean"])}
            for category, values in fraud_by_category.iterrows()
            if values["sum"] > 0
        },
    }


def run_fraud_analysis(
    mode: str,
    client_id: int | None = None,
    transaction_id: int | None = None,
    mcc_code: str | None = None,
    threshold: float = 500.0,
) -> dict:
    """Route fraud analysis requests through a single reusable entrypoint."""
    if mode == "transaction_check":
        if transaction_id is None:
            return {"error": "transaction_id is required for transaction_check"}
        return check_transaction_fraud(transaction_id)
    if mode == "client_summary":
        if client_id is None:
            return {"error": "client_id is required for client_summary"}
        return get_client_fraud_risk(client_id)
    if mode == "merchant_summary":
        if mcc_code is None:
            return {"error": "mcc_code is required for merchant_summary"}
        return get_merchant_fraud_summary(mcc_code)
    if mode == "high_risk_transactions":
        if client_id is None:
            return {"error": "client_id is required for high_risk_transactions"}
        return get_high_risk_transactions(client_id, threshold)
    if mode == "global_summary":
        return get_global_fraud_summary()
    return {"error": f"Unsupported fraud analysis mode '{mode}'"}
