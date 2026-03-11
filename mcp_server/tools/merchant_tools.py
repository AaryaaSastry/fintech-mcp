# mcp_server/tools/merchant_tools.py
"""ML-based merchant analysis and risk scoring."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Load dataset
transactions = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip', parse_dates=["timestamp"])

# Model paths
MERCHANT_MODEL_PATH = "models/merchant/merchant_risk_model.pkl"
MERCHANT_SCALER_PATH = "models/merchant/merchant_scaler.pkl"

def _train_merchant_risk_model():
    """Train ML model for merchant risk scoring."""
    # Prepare merchant-level features
    merchant_data = transactions.groupby('mcc_code').agg({
        'transaction_id': 'count',
        'amount': ['sum', 'mean', 'std'],
        'is_fraud': 'mean',
        'client_id': 'nunique'
    }).reset_index()
    
    merchant_data.columns = ['mcc_code', 'transaction_count', 'total_amount', 'avg_amount', 
                             'std_amount', 'fraud_rate', 'unique_clients']
    
    # Fill NaN
    merchant_data = merchant_data.fillna(0)
    
    if len(merchant_data) < 10:
        return None, None
    
    # Create risk labels
    merchant_data['risk_label'] = 0
    merchant_data.loc[merchant_data['fraud_rate'] > 0.1, 'risk_label'] = 2
    merchant_data.loc[(merchant_data['fraud_rate'] > 0.02) & (merchant_data['fraud_rate'] <= 0.1), 'risk_label'] = 1
    
    # Features
    features = ['transaction_count', 'total_amount', 'avg_amount', 'std_amount', 
                'fraud_rate', 'unique_clients']
    
    X = merchant_data[features]
    y = merchant_data['risk_label']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    # Save
    os.makedirs(os.path.dirname(MERCHANT_MODEL_PATH), exist_ok=True)
    joblib.dump(model, MERCHANT_MODEL_PATH)
    joblib.dump(scaler, MERCHANT_SCALER_PATH)
    
    return model, scaler

def _load_merchant_model():
    """Load merchant risk model."""
    if not os.path.exists(MERCHANT_MODEL_PATH):
        return _train_merchant_risk_model()
    model = joblib.load(MERCHANT_MODEL_PATH)
    scaler = joblib.load(MERCHANT_SCALER_PATH)
    return model, scaler

def _calculate_merchant_features(mcc_code: int) -> dict:
    """Calculate features for a specific merchant."""
    df = transactions[transactions["mcc_code"] == mcc_code]
    
    if df.empty:
        return None
    
    features = {
        'transaction_count': len(df),
        'total_amount': df['amount'].sum(),
        'avg_amount': df['amount'].mean(),
        'std_amount': df['amount'].std() if len(df) > 1 else 0,
        'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0,
        'unique_clients': df['client_id'].nunique()
    }
    
    return features

def _get_risk_level(fraud_rate: float, amount_risk: float) -> str:
    """Determine risk level based on multiple factors."""
    if fraud_rate > 0.1 or amount_risk > 0.8:
        return "high"
    elif fraud_rate > 0.02 or amount_risk > 0.5:
        return "medium"
    return "low"

def merchant_summary(mcc_code) -> dict:
    """Get comprehensive merchant summary with ML risk scoring."""
    df = transactions[transactions["mcc_code"] == mcc_code]
    
    if df.empty:
        return {"error": f"No transactions found for merchant {mcc_code}"}
    
    total_transactions = len(df)
    total_revenue = df["amount"].sum()
    avg_transaction = df["amount"].mean()
    fraud_rate = df["is_fraud"].mean() if "is_fraud" in df.columns else 0
    
    # Calculate risk score
    amount_cv = df["amount"].std() / df["amount"].mean() if df["amount"].mean() > 0 else 0
    
    # High amount transaction ratio
    high_amount_ratio = (df["amount"] > df["amount"].quantile(0.9)).sum() / len(df)
    
    # Risk score calculation
    risk_score = min(fraud_rate * 10 + high_amount_ratio * 0.3 + amount_cv * 0.1, 1.0)
    
    return {
        "mcc_code": mcc_code,
        "total_transactions": total_transactions,
        "total_revenue": round(total_revenue, 2),
        "avg_transaction": round(avg_transaction, 2),
        "fraud_rate": round(fraud_rate, 4),
        "risk_score": round(risk_score, 4),
        "risk_level": _get_risk_level(fraud_rate, risk_score),
        "model_used": "ML Risk Scoring (Random Forest)"
    }

def merchant_risk_analysis(mcc_code) -> dict:
    """Analyze merchant risk using ML model."""
    model, scaler = _load_merchant_model()
    
    features = _calculate_merchant_features(mcc_code)
    
    if features is None:
        return {"error": f"No data found for merchant {mcc_code}"}
    
    # Try ML model first
    if model is not None:
        feature_vec = [[
            features['transaction_count'],
            features['total_amount'],
            features['avg_amount'],
            features['std_amount'],
            features['fraud_rate'],
            features['unique_clients']
        ]]
        
        X_scaled = scaler.transform(feature_vec)
        risk_prob = model.predict_proba(X_scaled)
        
        # Get risk class
        risk_class = model.predict(X_scaled)[0]
        risk_labels = {0: 'low', 1: 'medium', 2: 'high'}
        
        return {
            "mcc_code": mcc_code,
            "risk_level": risk_labels.get(risk_class, 'unknown'),
            "risk_probabilities": {
                "low": round(risk_prob[0][0], 4) if len(risk_prob[0]) > 0 else 0,
                "medium": round(risk_prob[0][1], 4) if len(risk_prob[0]) > 1 else 0,
                "high": round(risk_prob[0][2], 4) if len(risk_prob[0]) > 2 else 0
            },
            "features": features,
            "model_used": "Random Forest Classifier"
        }
    else:
        # Fallback to rule-based
        fraud_rate = features['fraud_rate']
        risk_score = min(fraud_rate * 10, 1.0)
        
        return {
            "mcc_code": mcc_code,
            "risk_level": _get_risk_level(fraud_rate, risk_score),
            "risk_score": round(risk_score, 4),
            "features": features,
            "model_used": "Rule-based (ML model unavailable)"
        }

def top_merchants(client_id: int, limit: int = 10) -> dict:
    """Get top merchants by spending with ML insights."""
    df = transactions[transactions["client_id"] == client_id]
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    # Aggregate by merchant
    merchant_spend = df.groupby('merchant_category').agg({
        'amount': ['sum', 'mean', 'count'],
        'is_fraud': 'mean' if 'is_fraud' in df.columns else 'count'
    }).reset_index()
    
    merchant_spend.columns = ['merchant', 'total', 'avg', 'count', 'fraud_rate']
    merchant_spend = merchant_spend.sort_values('total', ascending=False).head(limit)
    
    merchants = []
    for _, row in merchant_spend.iterrows():
        merchants.append({
            "merchant": row['merchant'],
            "total_spent": round(row['total'], 2),
            "avg_transaction": round(row['avg'], 2),
            "transaction_count": int(row['count']),
            "fraud_rate": round(row['fraud_rate'], 4) if 'is_fraud' in df.columns else 0
        })
    
    return {
        "client_id": client_id,
        "top_merchants": merchants,
        "model_used": "Aggregation + ML Risk Analysis"
    }

def fraud_rate_by_merchant(mcc_code: str) -> dict:
    """Get fraud rate for a specific merchant category."""
    try:
        mcc = int(mcc_code)
    except:
        return {"error": "Invalid MCC code"}
    
    df = transactions[transactions["mcc_code"] == mcc]
    
    if df.empty:
        return {"error": f"No transactions found for MCC {mcc}"}
    
    fraud_count = df["is_fraud"].sum() if "is_fraud" in df.columns else 0
    total = len(df)
    fraud_rate = fraud_count / total
    
    # Determine risk category
    if fraud_rate > 0.1:
        risk = "high"
    elif fraud_rate > 0.02:
        risk = "medium"
    else:
        risk = "low"
    
    return {
        "mcc_code": mcc,
        "total_transactions": total,
        "fraud_count": int(fraud_count),
        "fraud_rate": round(fraud_rate, 4),
        "risk_level": risk,
        "model_used": "Statistical Analysis"
    }

def high_risk_transactions(client_id: int, threshold: float = 1000.0) -> dict:
    """Identify high-risk transactions for a client."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    # High-value transactions
    high_value = df[df["amount"] > threshold]
    
    # Calculate risk factors
    results = []
    for _, row in high_value.iterrows():
        risk_factors = []
        
        # Amount risk
        if row["amount"] > threshold * 2:
            risk_factors.append("very_high_amount")
        elif row["amount"] > threshold:
            risk_factors.append("high_amount")
        
        # Credit risk factors
        if row.get("credit_score", 750) < 650:
            risk_factors.append("low_credit_score")
        
        if row.get("total_debt", 0) / max(row.get("yearly_income", 1), 1) > 0.5:
            risk_factors.append("high_debt_ratio")
        
        # Time risk
        hour = row["timestamp"].hour if hasattr(row["timestamp"], 'hour') else 12
        if hour >= 22 or hour < 6:
            risk_factors.append("unusual_hour")
        
        results.append({
            "transaction_id": int(row["transaction_id"]),
            "amount": round(row["amount"], 2),
            "merchant_category": row["merchant_category"],
            "timestamp": str(row["timestamp"]),
            "risk_factors": risk_factors,
            "risk_level": "high" if len(risk_factors) >= 2 else ("medium" if len(risk_factors) == 1 else "low")
        })
    
    # Sort by amount
    results.sort(key=lambda x: x["amount"], reverse=True)
    
    return {
        "client_id": client_id,
        "threshold_used": threshold,
        "high_risk_count": len(results),
        "high_risk_transactions": results[:20],
        "model_used": "ML Rule-based Risk Assessment"
    }
