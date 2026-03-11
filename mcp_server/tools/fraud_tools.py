# mcp_server/tools/fraud_tools.py
"""ML-based fraud detection using Random Forest classifier."""
import pandas as pd
import joblib
import os
import numpy as np

# Load transaction data (C engine is faster; still tolerate bad lines)
transactions = pd.read_csv("data/transactions.csv", on_bad_lines='skip')
transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])

# Ensure numeric columns are properly typed
transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce').fillna(0)
transactions['credit_score'] = pd.to_numeric(transactions['credit_score'], errors='coerce').fillna(700)
transactions['yearly_income'] = pd.to_numeric(transactions['yearly_income'], errors='coerce').fillna(50000)
transactions['total_debt'] = pd.to_numeric(transactions['total_debt'], errors='coerce').fillna(0)
transactions['credit_limit'] = pd.to_numeric(transactions['credit_limit'], errors='coerce').fillna(10000)
transactions['current_age'] = pd.to_numeric(transactions['current_age'], errors='coerce').fillna(30)
transactions['mcc_code'] = pd.to_numeric(transactions['mcc_code'], errors='coerce').fillna(5411)
transactions['use_chip'] = transactions['use_chip'].astype(str).str.lower().isin(['true', '1', 'yes'])

# Model paths
MODEL_PATH = "models/fraud/fraud_model.pkl"
SCALER_PATH = "models/fraud/scaler.pkl"

# Cache the model at module level to avoid reloading on every call
_model_cache = None
_scaler_cache = None

def _load_model():
    """Load the trained ML model and scaler (cached)."""
    global _model_cache, _scaler_cache
    
    # Return cached model if already loaded
    if _model_cache is not None and _scaler_cache is not None:
        return _model_cache, _scaler_cache
    
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    _model_cache = joblib.load(MODEL_PATH)
    # Avoid Windows process-spawn overhead / permission issues during predict
    try:
        _model_cache.n_jobs = 1
    except Exception:
        pass
    _scaler_cache = joblib.load(SCALER_PATH)
    return _model_cache, _scaler_cache

def _build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature builder (keeps column order identical to the old row-based path)."""
    features = pd.DataFrame(index=df.index)
    features['amount'] = df['amount']
    features['amount_log'] = np.log1p(df['amount'])
    features['credit_score'] = df['credit_score']
    features['yearly_income'] = df['yearly_income']
    features['total_debt'] = df['total_debt']
    safe_income = df['yearly_income'].replace(0, 1)
    features['debt_to_income'] = df['total_debt'] / safe_income
    features['credit_limit'] = df['credit_limit']
    features['utilization'] = df['credit_limit'] / safe_income
    features['current_age'] = df['current_age']
    features['age_group'] = pd.cut(df['current_age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    features['use_chip'] = df['use_chip'].astype(int)
    card_type_map = {'Credit': 1, 'Debit': 0, 'Prepaid': 2}
    features['card_type_encoded'] = df['card_type'].map(card_type_map).fillna(0).astype(int)
    card_brand_map = {'Visa': 0, 'Mastercard': 1, 'Discover': 2, 'Amex': 3}
    features['card_brand_encoded'] = df['card_brand'].map(card_brand_map).fillna(0).astype(int)
    features['hour'] = df['timestamp'].dt.hour
    features['day_of_week'] = df['timestamp'].dt.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['is_night'] = ((features['hour'] >= 22) | (features['hour'] < 6)).astype(int)
    features['mcc_code'] = df['mcc_code']
    features['high_amount'] = (df['amount'] > 1000).astype(int)
    features['low_credit_score'] = (df['credit_score'] < 650).astype(int)
    features['high_debt_ratio'] = (features['debt_to_income'] > 0.5).astype(int)
    return features

def _get_risk_level(prob):
    """Convert probability to risk level."""
    if prob > 0.5:
        return 'high'
    elif prob > 0.2:
        return 'medium'
    return 'low'

def check_fraud(transaction_id: int) -> dict:
    """Check fraud probability for a specific transaction using ML model."""
    df = transactions[transactions["transaction_id"] == transaction_id]
    
    if df.empty:
        return {"error": "Transaction not found"}
    
    model, scaler = _load_model()
    
    if model is not None:
        # Use ML model
        features = _build_features_df(df)
        X = scaler.transform(features)
        fraud_prob = model.predict_proba(X)[0][1]
        is_fraud = bool(model.predict(X)[0])
        
        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": is_fraud,
            "risk_level": _get_risk_level(fraud_prob),
            "model_used": "Random Forest ML"
        }
    else:
        # Fallback to heuristic
        prob = 0.05
        row = df.iloc[0]
        if row["amount"] > 1000:
            prob += 0.2
        if row["credit_score"] < 600:
            prob += 0.1
        if row["total_debt"] / max(row["yearly_income"], 1) > 0.5:
            prob += 0.1
        
        prob = min(prob, 1.0)
        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(prob, 2),
            "is_fraud": prob > 0.5,
            "risk_level": _get_risk_level(prob),
            "model_used": "Heuristic (ML not trained)"
        }

def get_client_fraud_risk(client_id: int) -> dict:
    """Get overall fraud risk for a client based on their transaction history."""
    client_transactions = transactions[transactions["client_id"] == client_id]
    
    if client_transactions.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    model, scaler = _load_model()
    
    if model is not None:
        features = _build_features_df(client_transactions)
        X = scaler.transform(features)
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        total_risk = float(probs.sum())
        fraud_transactions = int(preds.sum())
    else:
        # Heuristic fallback (vectorized)
        prob = np.full(len(client_transactions), 0.05)
        prob += (client_transactions["amount"] > 1000) * 0.2
        prob += (client_transactions["credit_score"] < 600) * 0.1
        prob += (client_transactions["total_debt"] / client_transactions["yearly_income"].replace(0, 1) > 0.5) * 0.1
        prob = np.clip(prob, 0, 1)
        total_risk = float(prob.sum())
        fraud_transactions = int((prob > 0.5).sum())
        probs = prob

    high_value_count = int((client_transactions['amount'] > 1000).sum())
    avg_risk = total_risk / len(client_transactions)
    
    return {
        "client_id": client_id,
        "risk_score": round(avg_risk, 4),
        "high_value_transactions": high_value_count,
        "total_transactions": len(client_transactions),
        "fraud_transactions_detected": fraud_transactions,
        "risk_level": _get_risk_level(avg_risk),
        "model_used": "Random Forest ML" if model else "Heuristic"
    }

def detect_anomalies() -> dict:
    """Detect potentially fraudulent transactions across all data."""
    model, scaler = _load_model()
    
    high_risk = []
    
    if model is not None:
        features = _build_features_df(transactions)
        X = scaler.transform(features)
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        mask = probs > 0.3
        for tx_row, prob, is_fraud in zip(transactions[mask].itertuples(index=False), probs[mask], preds[mask]):
            high_risk.append({
                "transaction_id": int(tx_row.transaction_id),
                "amount": float(tx_row.amount),
                "fraud_probability": round(float(prob), 4),
                "is_fraud": bool(is_fraud),
                "merchant_category": tx_row.merchant_category
            })
    else:
        # Heuristic
        prob = np.full(len(transactions), 0.05)
        prob += (transactions["amount"] > 1000) * 0.2
        prob += (transactions["credit_score"] < 600) * 0.1
        prob = np.clip(prob, 0, 1)
        mask = prob > 0.3
        for tx_row, p in zip(transactions[mask].itertuples(index=False), prob[mask]):
            high_risk.append({
                "transaction_id": int(tx_row.transaction_id),
                "amount": float(tx_row.amount),
                "fraud_probability": round(float(p), 2),
                "is_fraud": p > 0.5,
                "merchant_category": tx_row.merchant_category
            })
    
    # Sort by fraud probability
    high_risk.sort(key=lambda x: x['fraud_probability'], reverse=True)
    
    return {
        "total_anomalies": len(high_risk),
        "high_risk_transactions": high_risk[:10],  # Top 10
        "model_used": "Random Forest ML" if model else "Heuristic"
    }
