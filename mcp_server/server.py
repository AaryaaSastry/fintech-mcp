"""
Modular MCP Server with ALL Analytics Tools
==========================================
This server provides 25+ analytical tools for fintech data.
Each tool is modular and can be easily extended.
"""
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import pandas as pd
import pathlib
import os

# Initialize MCP server
mcp = FastMCP("FintechAnalytics", log_level="ERROR")

# ===== DATA LOADING =====
DATA_PATH = pathlib.Path("data/transactions.csv")

def load_data():
    """Load and preprocess transaction data."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(
        DATA_PATH,
        on_bad_lines='skip',
        parse_dates=["timestamp"]
    )
    
    # Ensure correct types
    numeric_cols = ['transaction_id', 'client_id', 'amount', 'card_id', 'mcc_code', 
                   'credit_limit', 'is_fraud', 'current_age', 'yearly_income', 
                   'total_debt', 'credit_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load data once at startup
transactions = load_data()

# Helper function for error responses
def error_response(msg: str):
    return {"error": msg}

# ============================================================
# SPENDING TOOLS
# ============================================================

@mcp.tool(name="get_spending_summary", description="Get total spending and top merchant categories for a client over the last N months.")
def get_spending_summary(client_id: int = Field(), months: int = Field(default=3)):
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    last_date = df["timestamp"].max()
    df = df[df["timestamp"] >= (last_date - pd.Timedelta(days=months*30))]
    
    total = float(df["amount"].sum())
    top_merchants = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(5)
    top_merchants = {k: float(v) for k, v in top_merchants.items()}
    
    return {"total_spent": total, "top_merchants": top_merchants, "transaction_count": len(df)}

@mcp.tool(name="spending_by_category", description="Breakdown of spending by merchant category.")
def spending_by_category(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    cat_spend = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=False)
    return {"categories": {k: float(v) for k, v in cat_spend.items()}}

@mcp.tool(name="monthly_spending_trend", description="Monthly spending trend over time.")
def monthly_spending_trend(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()
    return {"months": {str(k): float(v) for k, v in monthly.items()}}

@mcp.tool(name="spending_by_city", description="Spending breakdown by merchant city.")
def spending_by_city(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    city_spend = df.groupby("merchant_city")["amount"].sum().sort_values(ascending=False).head(10)
    return {"cities": {k: float(v) for k, v in city_spend.items()}}

@mcp.tool(name="spending_by_card_type", description="Spending by card type (Credit/Debit/Prepaid).")
def spending_by_card_type(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    card_spend = df.groupby("card_type")["amount"].sum().sort_values(ascending=False)
    return {"card_types": {k: float(v) for k, v in card_spend.items()}}

@mcp.tool(name="total_transactions_count", description="Total number of transactions.")
def total_transactions_count(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    return {"total_transactions": len(df), "client_id": client_id}

@mcp.tool(name="average_transaction_amount", description="Average transaction amount.")
def average_transaction_amount(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    return {"average_amount": float(df["amount"].mean()), "client_id": client_id}

# ============================================================
# FRAUD TOOLS - Using ML Model
# ============================================================

# Import ML model
import joblib
import numpy as np

ML_MODEL_PATH = "models/fraud/fraud_model.pkl"
ML_SCALER_PATH = "models/fraud/scaler.pkl"

def _load_ml_model():
    """Load trained ML model and scaler."""
    try:
        if os.path.exists(ML_MODEL_PATH):
            model = joblib.load(ML_MODEL_PATH)
            # Prevent joblib from spawning processes on predict (Windows WinError 5)
            try:
                model.n_jobs = 1
            except Exception:
                pass
            return model, joblib.load(ML_SCALER_PATH)
    except:
        pass
    return None, None

def _build_ml_features(row):
    """Build feature vector for ML model."""
    features = {}
    features['amount'] = float(row.get('amount', 0))
    features['amount_log'] = np.log1p(float(row.get('amount', 0)))
    features['credit_score'] = float(row.get('credit_score', 700))
    features['yearly_income'] = float(row.get('yearly_income', 50000))
    features['total_debt'] = float(row.get('total_debt', 0))
    features['debt_to_income'] = float(row.get('total_debt', 0)) / max(float(row.get('yearly_income', 1)), 1)
    features['credit_limit'] = float(row.get('credit_limit', 10000))
    features['utilization'] = float(row.get('credit_limit', 10000)) / max(float(row.get('yearly_income', 1)), 1)
    features['current_age'] = float(row.get('current_age', 30))
    age = float(row.get('current_age', 30))
    if age <= 25: features['age_group'] = 1
    elif age <= 35: features['age_group'] = 2
    elif age <= 45: features['age_group'] = 3
    elif age <= 55: features['age_group'] = 4
    else: features['age_group'] = 5
    features['use_chip'] = int(bool(row.get('use_chip', False)))
    card_type = str(row.get('card_type', ''))
    features['card_type_encoded'] = {'Credit': 1, 'Debit': 0, 'Prepaid': 2}.get(card_type, 0)
    card_brand = str(row.get('card_brand', ''))
    features['card_brand_encoded'] = {'Visa': 0, 'Mastercard': 1, 'Discover': 2, 'Amex': 3}.get(card_brand, 0)
    ts = row.get('timestamp', pd.Timestamp.now())
    if isinstance(ts, str): ts = pd.to_datetime(ts)
    features['hour'] = ts.hour if hasattr(ts, 'hour') else 12
    features['day_of_week'] = ts.dayofweek if hasattr(ts, 'dayofweek') else 0
    features['is_weekend'] = 1 if (hasattr(ts, 'dayofweek') and ts.dayofweek >= 5) else 0
    features['is_night'] = 1 if (hasattr(ts, 'hour') and (ts.hour >= 22 or ts.hour < 6)) else 0
    features['mcc_code'] = float(row.get('mcc_code', 5411))
    features['high_amount'] = 1 if float(row.get('amount', 0)) > 1000 else 0
    features['low_credit_score'] = 1 if float(row.get('credit_score', 700)) < 650 else 0
    features['high_debt_ratio'] = 1 if features['debt_to_income'] > 0.5 else 0
    return features

def _get_risk_level(prob):
    """Convert probability to risk level."""
    if prob > 0.5: return 'high'
    elif prob > 0.2: return 'medium'
    return 'low'

@mcp.tool(name="check_fraud", description="Estimate fraud probability for a transaction using ML model.")
def check_fraud(transaction_id: int = Field()):
    df = transactions[transactions["transaction_id"] == transaction_id]
    if df.empty:
        return error_response(f"Transaction {transaction_id} not found")
    
    row = df.iloc[0]
    model, scaler = _load_ml_model()
    
    if model is not None:
        # Use ML model
        features = _build_ml_features(row)
        feature_df = pd.DataFrame([features])
        X = scaler.transform(feature_df)
        prob = model.predict_proba(X)[0][1]
        is_fraud = bool(model.predict(X)[0])
        
        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(float(prob), 4),
            "is_fraud": is_fraud,
            "risk_level": _get_risk_level(prob),
            "model_used": "Random Forest ML"
        }
    else:
        # Fallback to heuristic
        prob = 0.05
        if row.get("amount", 0) > 1000: prob += 0.2
        if row.get("credit_score", 700) < 600: prob += 0.1
        if row.get("yearly_income", 1) > 0:
            debt_ratio = row.get("total_debt", 0) / row["yearly_income"]
            if debt_ratio > 0.5: prob += 0.1
        
        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(min(prob, 1.0), 2),
            "risk_level": _get_risk_level(prob),
            "model_used": "Heuristic (ML not available)"
        }

@mcp.tool(name="client_fraud_risk", description="Calculate overall fraud risk score for a client using ML model.")
def client_fraud_risk(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    model, scaler = _load_ml_model()
    
    total_risk = 0
    high_value_count = 0
    fraud_transactions = 0
    
    for _, row in df.iterrows():
        if model is not None:
            try:
                features = _build_ml_features(row)
                feature_df = pd.DataFrame([features])
                X = scaler.transform(feature_df)
                prob = model.predict_proba(X)[0][1]
                is_fraud = model.predict(X)[0]
                
                total_risk += prob
                if row['amount'] > 1000:
                    high_value_count += 1
                if is_fraud:
                    fraud_transactions += 1
            except:
                # Fallback heuristic
                prob = 0.05
                if row.get("amount", 0) > 1000:
                    prob += 0.2
                    high_value_count += 1
                if row.get("credit_score", 700) < 600:
                    prob += 0.1
                if row.get("yearly_income", 1) > 0:
                    if row.get("total_debt", 0) / row["yearly_income"] > 0.5:
                        prob += 0.1
                total_risk += min(prob, 1.0)
                if prob > 0.5:
                    fraud_transactions += 1
        else:
            # Heuristic fallback
            prob = 0.05
            if row.get("amount", 0) > 1000:
                prob += 0.2
                high_value_count += 1
            if row.get("credit_score", 700) < 600:
                prob += 0.1
            if row.get("yearly_income", 1) > 0:
                if row.get("total_debt", 0) / row["yearly_income"] > 0.5:
                    prob += 0.1
            total_risk += min(prob, 1.0)
            if prob > 0.5:
                fraud_transactions += 1
    
    avg_risk = total_risk / len(df)
    
    return {
        "client_id": client_id,
        "risk_score": round(float(avg_risk), 4),
        "high_value_transactions": high_value_count,
        "total_transactions": len(df),
        "fraud_transactions_detected": fraud_transactions,
        "risk_level": _get_risk_level(avg_risk),
        "model_used": "Random Forest ML" if model else "Heuristic"
    }

@mcp.tool(name="fraud_rate_by_merchant", description="Fraud rate by merchant category.")
def fraud_rate_by_merchant(mcc_code: str = Field()):
    df = transactions[transactions["mcc_code"].astype(str) == mcc_code]
    if df.empty:
        return error_response(f"No data for MCC {mcc_code}")
    
    fraud_rate = df["is_fraud"].mean() if "is_fraud" in df.columns else 0
    return {"mcc_code": mcc_code, "fraud_rate": round(float(fraud_rate), 3), "total_transactions": len(df)}

@mcp.tool(name="high_risk_transactions", description="List high-value transactions above threshold.")
def high_risk_transactions(client_id: int = Field(), threshold: float = Field(default=500)):
    df = transactions[(transactions["client_id"] == client_id) & (transactions["amount"] > threshold)]
    if df.empty:
        return {"transactions": [], "message": f"No transactions above ${threshold}"}
    
    result = df[["transaction_id", "amount", "merchant_category", "timestamp"]].to_dict("records")
    return {"transactions": result, "count": len(result)}

# ============================================================
# CREDIT TOOLS
# ============================================================

@mcp.tool(name="credit_score_history", description="Credit score with grade.")
def credit_score_history(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No data for client {client_id}")
    
    score = int(df.iloc[0].get("credit_score", 0))
    grade = "excellent" if score >= 750 else "good" if score >= 700 else "fair" if score >= 650 else "poor"
    
    return {"client_id": client_id, "credit_score": score, "grade": grade}

@mcp.tool(name="debt_income_analysis", description="Debt to income ratio analysis.")
def debt_income_analysis(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No data for client {client_id}")
    
    row = df.iloc[0]
    debt = float(row.get("total_debt", 0))
    income = float(row.get("yearly_income", 1))
    ratio = debt / max(income, 1)
    
    return {
        "client_id": client_id,
        "total_debt": debt,
        "yearly_income": income,
        "debt_to_income_ratio": round(ratio, 3),
        "risk_level": "high" if ratio > 0.5 else "medium" if ratio > 0.3 else "low"
    }

@mcp.tool(name="credit_limit_analysis", description="Credit limit and utilization analysis.")
def credit_limit_analysis(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No data for client {client_id}")
    
    row = df.iloc[0]
    limit = float(row.get("credit_limit", 0))
    spent = float(df["amount"].sum())
    utilization = (spent / max(limit, 1)) * 100
    
    return {
        "client_id": client_id,
        "credit_limit": limit,
        "total_spent": spent,
        "utilization_percent": round(utilization, 2),
        "status": "over_limit" if utilization > 100 else "high" if utilization > 75 else "normal"
    }

# ============================================================
# FORECAST TOOLS
# ============================================================

@mcp.tool(name="spending_forecast", description="Forecast spending for next N months.")
def spending_forecast(client_id: int = Field(), months: int = Field(default=3)):
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()
    avg = float(monthly.mean())
    
    forecast = {f"month_{i+1}": round(avg, 2) for i in range(months)}
    return {"client_id": client_id, "forecast": forecast, "average_monthly": round(avg, 2)}

# ============================================================
# VISUALIZATION TOOLS
# ============================================================

@mcp.tool(name="transaction_heatmap", description="Heatmap of spending by day/hour.")
def transaction_heatmap(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    heatmap = df.groupby(["day_of_week", "hour"])["amount"].sum().unstack(fill_value=0)
    
    result = {}
    for day in heatmap.index:
        result[int(day)] = {int(h): float(v) for h, v in heatmap.loc[day].items()}
    
    return {"heatmap": result}

@mcp.tool(name="peak_hours_analysis", description="Analyze peak transaction hours and days.")
def peak_hours_analysis(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    
    peak_hours = df.groupby("hour")["amount"].sum().sort_values(ascending=False).head(5)
    peak_days = df.groupby("day_name")["amount"].sum().sort_values(ascending=False)
    
    return {
        "peak_hours": {int(k): float(v) for k, v in peak_hours.items()},
        "peak_days": {k: float(v) for k, v in peak_days.items()}
    }

@mcp.tool(name="spending_distribution", description="Distribution of transaction amounts.")
def spending_distribution(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    amounts = df["amount"].describe()
    return {
        "count": int(amounts["count"]),
        "min": float(amounts["min"]),
        "max": float(amounts["max"]),
        "mean": float(amounts["mean"]),
        "median": float(amounts["50%"]),
        "std": float(amounts["std"])
    }

# ============================================================
# MERCHANT TOOLS
# ============================================================

@mcp.tool(name="merchant_summary", description="Merchant summary by MCC code.")
def merchant_summary(mcc_code: str = Field()):
    df = transactions[transactions["mcc_code"].astype(str) == mcc_code]
    if df.empty:
        return error_response(f"No data for MCC {mcc_code}")
    
    return {
        "mcc_code": mcc_code,
        "total_transactions": len(df),
        "total_revenue": float(df["amount"].sum()),
        "avg_transaction": float(df["amount"].mean()),
        "fraud_rate": float(df["is_fraud"].mean()) if "is_fraud" in df.columns else 0
    }

@mcp.tool(name="merchant_risk_analysis", description="Analyze merchant risk level.")
def merchant_risk_analysis(mcc_code: str = Field()):
    df = transactions[transactions["mcc_code"].astype(str) == mcc_code]
    if df.empty:
        return error_response(f"No data for MCC {mcc_code}")
    
    fraud_rate = df["is_fraud"].mean() if "is_fraud" in df.columns else 0
    avg_amount = df["amount"].mean()
    
    return {
        "mcc_code": mcc_code,
        "fraud_rate": round(float(fraud_rate), 3),
        "avg_amount": float(avg_amount),
        "risk_level": "high" if fraud_rate > 0.1 else "medium" if fraud_rate > 0.05 else "low"
    }

@mcp.tool(name="top_merchants", description="Top merchants by spending.")
def top_merchants(client_id: int = Field(), limit: int = Field(default=5)):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    top = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(limit)
    return {"merchants": {k: float(v) for k, v in top.items()}}

# ============================================================
# CARD TOOLS
# ============================================================

@mcp.tool(name="card_usage_analysis", description="Card usage patterns.")
def card_usage_analysis(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    chip_usage = df.groupby("use_chip")["amount"].sum()
    card_types = df.groupby("card_type")["amount"].sum()
    card_brands = df.groupby("card_brand")["amount"].sum()
    
    return {
        "chip_vs_swipe": {str(k): float(v) for k, v in chip_usage.items()},
        "by_card_type": {k: float(v) for k, v in card_types.items()},
        "by_card_brand": {k: float(v) for k, v in card_brands.items()}
    }

@mcp.tool(name="card_type_spending", description="Spending by card brand.")
def card_type_spending(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No transactions for client {client_id}")
    
    brands = df.groupby("card_brand")["amount"].sum().sort_values(ascending=False)
    return {"brands": {k: float(v) for k, v in brands.items()}}

# ============================================================
# PROFILE TOOLS
# ============================================================

@mcp.tool(name="customer_profile", description="Complete customer profile.")
def customer_profile(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No data for client {client_id}")
    
    row = df.iloc[0]
    return {
        "client_id": client_id,
        "current_age": int(row.get("current_age", 0)),
        "yearly_income": float(row.get("yearly_income", 0)),
        "total_debt": float(row.get("total_debt", 0)),
        "credit_score": int(row.get("credit_score", 0)),
        "credit_limit": float(row.get("credit_limit", 0)),
        "total_transactions": len(df),
        "total_spent": float(df["amount"].sum())
    }

@mcp.tool(name="income_analysis", description="Income and spending ratio analysis.")
def income_analysis(client_id: int = Field()):
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return error_response(f"No data for client {client_id}")
    
    row = df.iloc[0]
    income = float(row.get("yearly_income", 1))
    spent = float(df["amount"].sum())
    ratio = spent / max(income, 1)
    
    return {
        "client_id": client_id,
        "yearly_income": income,
        "total_spent": spent,
        "spending_to_income_ratio": round(ratio, 3),
        "status": "high" if ratio > 0.5 else "moderate" if ratio > 0.25 else "low"
    }

# ============================================================
# ALL-USERS AGGREGATE TOOLS (No client_id required)
# ============================================================

@mcp.tool(name="get_all_users_spending_summary", description="Get spending summary for ALL users/clients combined.")
def get_all_users_spending_summary(months: int = Field(default=3)):
    df = transactions.copy()
    if df.empty:
        return error_response("No transaction data available")
    
    last_date = df["timestamp"].max()
    df = df[df["timestamp"] >= (last_date - pd.Timedelta(days=months*30))]
    
    total = float(df["amount"].sum())
    unique_clients = df["client_id"].nunique()
    transaction_count = len(df)
    avg_per_transaction = float(df["amount"].mean())
    avg_per_client = total / unique_clients if unique_clients > 0 else 0
    
    top_merchants = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(5)
    top_merchants = {k: float(v) for k, v in top_merchants.items()}
    
    # Get individual client breakdown
    client_breakdown = df.groupby("client_id").agg({
        "amount": ["sum", "mean", "count"],
        "yearly_income": "first",
        "credit_score": "first",
        "current_age": "first"
    }).reset_index()
    client_breakdown.columns = ["client_id", "total_spent", "avg_transaction", "transaction_count", "yearly_income", "credit_score", "age"]
    
    # Client name mapping
    client_name_map = {
        5: "Soniya", 12: "John", 15: "Emma", 22: "Mike", 8: "Sarah",
        1: "Alice", 2: "Bob", 3: "Charlie", 4: "David", 6: "Frank",
        7: "Grace", 9: "Henry", 10: "Ivy", 11: "Jack", 13: "Kate",
        14: "Leo", 16: "Maria", 17: "Nathan", 18: "Olivia", 19: "Peter"
    }
    
    # Build individual client details
    client_details = []
    for _, row in client_breakdown.iterrows():
        cid = int(row['client_id'])
        client_details.append({
            "client_id": cid,
            "name": client_name_map.get(cid, f"Client_{cid}"),
            "total_spent": round(float(row['total_spent']), 2),
            "avg_transaction": round(float(row['avg_transaction']), 2),
            "transaction_count": int(row['transaction_count']),
            "yearly_income": float(row['yearly_income']) if pd.notna(row['yearly_income']) else 0,
            "credit_score": int(row['credit_score']) if pd.notna(row['credit_score']) else 0,
            "age": int(row['age']) if pd.notna(row['age']) else 0
        })
    
    # Sort by total_spent descending
    client_details.sort(key=lambda x: x['total_spent'], reverse=True)
    
    return {
        "total_spent": total,
        "unique_clients": int(unique_clients),
        "transaction_count": transaction_count,
        "avg_per_transaction": round(avg_per_transaction, 2),
        "avg_per_client": round(avg_per_client, 2),
        "top_merchants": top_merchants,
        "months": months,
        "client_details": client_details
    }

@mcp.tool(name="spending_by_category_all", description="Spending breakdown by merchant category for ALL users.")
def spending_by_category_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    cat_spend = df.groupby("merchant_category")["amount"].agg(['sum', 'mean', 'count'])
    cat_spend = cat_spend.sort_values('sum', ascending=False)
    
    categories = {}
    for cat, row in cat_spend.iterrows():
        categories[cat] = {
            "total": float(row['sum']),
            "avg": float(row['mean']),
            "count": int(row['count'])
        }
    
    return {"categories": categories, "total_unique_categories": len(categories)}

@mcp.tool(name="monthly_spending_trend_all", description="Monthly spending trend for ALL users.")
def monthly_spending_trend_all():
    df = transactions.copy()
    if df.empty:
        return error_response("No data available")
    
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].agg(['sum', 'mean', 'count'])
    monthly = monthly.sort_index()
    
    return {
        "months": {str(k): {
            "total": float(v['sum']),
            "avg": float(v['mean']),
            "count": int(v['count'])
        } for k, v in monthly.iterrows()}
    }

@mcp.tool(name="spending_by_city_all", description="Spending breakdown by city for ALL users.")
def spending_by_city_all():
    df = transactions
    if df.empty or 'merchant_city' not in df.columns:
        return error_response("No city data available")
    
    city_spend = df.groupby("merchant_city")["amount"].agg(['sum', 'mean', 'count'])
    city_spend = city_spend.sort_values('sum', ascending=False).head(20)
    
    cities = {}
    for city, row in city_spend.iterrows():
        cities[city] = {
            "total": float(row['sum']),
            "avg": float(row['mean']),
            "count": int(row['count'])
        }
    
    return {"cities": cities, "total_unique_cities": len(cities)}

@mcp.tool(name="spending_by_card_type_all", description="Spending by card type for ALL users.")
def spending_by_card_type_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    card_spend = df.groupby("card_type")["amount"].agg(['sum', 'mean', 'count'])
    card_spend = card_spend.sort_values('sum', ascending=False)
    
    cards = {}
    for card, row in card_spend.iterrows():
        cards[card] = {
            "total": float(row['sum']),
            "avg": float(row['mean']),
            "count": int(row['count'])
        }
    
    return {"card_types": cards}

@mcp.tool(name="total_transactions_all", description="Total transactions count for ALL users.")
def total_transactions_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    return {
        "total_transactions": len(df),
        "unique_clients": int(df["client_id"].nunique()),
        "unique_merchants": int(df["merchant_category"].nunique())
    }

@mcp.tool(name="average_transaction_all", description="Average transaction amount for ALL users.")
def average_transaction_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    amounts = df["amount"]
    return {
        "average": float(amounts.mean()),
        "median": float(amounts.median()),
        "min": float(amounts.min()),
        "max": float(amounts.max()),
        "std": float(amounts.std()),
        "total_transactions": len(df)
    }

# Cache for aggregate functions (60 second TTL)
_aggregate_cache = {}
_CACHE_TTL = 60

def _get_cached(key, func):
    """Get cached result or compute new one."""
    import time
    now = time.time()
    if key in _aggregate_cache:
        cached_time, cached_result = _aggregate_cache[key]
        if now - cached_time < _CACHE_TTL:
            return cached_result
    result = func()
    _aggregate_cache[key] = (now, result)
    return result

@mcp.tool(name="global_fraud_analysis", description="Global fraud analysis for ALL transactions.")
def global_fraud_analysis():
    return _get_cached("global_fraud_analysis", _compute_global_fraud)

def _compute_global_fraud():
    df = transactions
    if df.empty or "is_fraud" not in df.columns:
        return error_response("No fraud data available")
    
    total = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = fraud_count / total if total > 0 else 0
    
    fraud_by_category = df.groupby("merchant_category")["is_fraud"].agg(['sum', 'mean'])
    fraud_by_category = {cat: {"count": int(row['sum']), "rate": float(row['mean'])} 
                         for cat, row in fraud_by_category.iterrows() if row['sum'] > 0}
    
    fraud_by_card = df.groupby("card_type")["is_fraud"].agg(['sum', 'mean'])
    fraud_by_card = {cat: {"count": int(row['sum']), "rate": float(row['mean'])} 
                     for cat, row in fraud_by_card.iterrows() if row['sum'] > 0}
    
    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_rate": round(fraud_rate, 4),
        "fraud_by_category": fraud_by_category,
        "fraud_by_card_type": fraud_by_card
    }

@mcp.tool(name="all_clients_fraud_summary", description="Fraud summary for ALL clients.")
def all_clients_fraud_summary():
    return _get_cached("all_clients_fraud_summary", _compute_all_clients_fraud)

def _compute_all_clients_fraud():
    df = transactions
    if df.empty or "is_fraud" not in df.columns:
        return error_response("No fraud data available")
    
    client_fraud = df.groupby("client_id").agg({
        "transaction_id": "count",
        "amount": "sum",
        "is_fraud": "sum"
    }).reset_index()
    
    client_fraud['fraud_rate'] = client_fraud['is_fraud'] / client_fraud['transaction_id']
    top_fraudulent = client_fraud.nlargest(10, 'fraud_rate')
    
    return {
        "total_clients": int(df["client_id"].nunique()),
        "clients_with_fraud": int((client_fraud['is_fraud'] > 0).sum()),
        "top_fraudulent_clients": [
            {"client_id": int(row['client_id']), "fraud_rate": round(float(row['fraud_rate']), 4), "fraud_count": int(row['is_fraud'])}
            for _, row in top_fraudulent.iterrows()
        ]
    }

@mcp.tool(name="credit_score_distribution_all", description="Credit score distribution for ALL users.")
def credit_score_distribution_all():
    df = transactions
    if df.empty or "credit_score" not in df.columns:
        return error_response("No credit score data available")
    
    scores = df.groupby("client_id")["credit_score"].first()
    
    return {
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "min": int(scores.min()),
        "max": int(scores.max()),
        "distribution": {
            "excellent_750+": int((scores >= 750).sum()),
            "good_700_749": int(((scores >= 700) & (scores < 750)).sum()),
            "fair_650_699": int(((scores >= 650) & (scores < 700)).sum()),
            "poor_below_650": int((scores < 650).sum())
        },
        "total_clients": len(scores)
    }

@mcp.tool(name="debt_income_all", description="Debt to income analysis for ALL users.")
def debt_income_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    client_data = df.groupby("client_id").agg({
        "yearly_income": "first",
        "total_debt": "first"
    }).reset_index()
    
    client_data['debt_to_income'] = client_data['total_debt'] / client_data['yearly_income'].replace(0, 1)
    
    return {
        "mean_debt_to_income": round(float(client_data['debt_to_income'].mean()), 3),
        "median_debt_to_income": round(float(client_data['debt_to_income'].median()), 3),
        "high_risk_clients": int((client_data['debt_to_income'] > 0.5).sum()),
        "medium_risk_clients": int(((client_data['debt_to_income'] > 0.3) & (client_data['debt_to_income'] <= 0.5)).sum()),
        "low_risk_clients": int((client_data['debt_to_income'] <= 0.3).sum()),
        "total_clients": len(client_data)
    }

@mcp.tool(name="credit_limit_all", description="Credit limit and utilization for ALL users.")
def credit_limit_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    client_data = df.groupby("client_id").agg({
        "credit_limit": "first",
        "amount": "sum"
    }).reset_index()
    
    client_data['utilization'] = (client_data['amount'] / client_data['credit_limit'].replace(0, 1)) * 100
    
    return {
        "mean_credit_limit": float(client_data['credit_limit'].mean()),
        "mean_utilization": round(float(client_data['utilization'].mean()), 2),
        "high_utilization_clients": int((client_data['utilization'] > 75).sum()),
        "over_limit_clients": int((client_data['utilization'] > 100).sum()),
        "total_clients": len(client_data)
    }

@mcp.tool(name="transaction_heatmap_all", description="Transaction heatmap for ALL users.")
def transaction_heatmap_all():
    df = transactions.copy()
    if df.empty:
        return error_response("No data available")
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    heatmap = df.groupby(["day_of_week", "hour"])["amount"].sum().unstack(fill_value=0)
    
    result = {}
    for day in heatmap.index:
        result[int(day)] = {int(h): float(v) for h, v in heatmap.loc[day].items()}
    
    peak_day_hour = df.groupby(["day_of_week", "hour"])["amount"].sum().idxmax()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    return {
        "heatmap": result,
        "peak_day": day_names[int(peak_day_hour[0])],
        "peak_hour": int(peak_day_hour[1]),
        "total_transactions": len(df)
    }

@mcp.tool(name="peak_hours_all", description="Peak transaction hours for ALL users.")
def peak_hours_all():
    df = transactions.copy()
    if df.empty:
        return error_response("No data available")
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    hourly = df.groupby("hour")["amount"].agg(['sum', 'mean', 'count'])
    daily = df.groupby("day_of_week")["amount"].agg(['sum', 'mean', 'count'])
    
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    return {
        "by_hour": {int(k): float(v['sum']) for k, v in hourly.iterrows()},
        "by_day": {day_names[int(k)]: float(v['sum']) for k, v in daily.iterrows()},
        "peak_hour": int(hourly['sum'].idxmax()),
        "peak_day": day_names[int(daily['sum'].idxmax())]
    }

@mcp.tool(name="spending_distribution_all", description="Spending distribution for ALL users.")
def spending_distribution_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    amounts = df["amount"]
    percentiles = {
        "25%": float(amounts.quantile(0.25)),
        "50%": float(amounts.quantile(0.50)),
        "75%": float(amounts.quantile(0.75)),
        "90%": float(amounts.quantile(0.90)),
        "95%": float(amounts.quantile(0.95)),
        "99%": float(amounts.quantile(0.99))
    }
    
    bins = [0, 20, 50, 100, 200, 500, 1000, float('inf')]
    labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
    df["bin"] = pd.cut(df["amount"], bins=bins, labels=labels)
    distribution = df["bin"].value_counts().sort_index().to_dict()
    
    return {
        "statistics": {
            "mean": float(amounts.mean()),
            "median": float(amounts.median()),
            "std": float(amounts.std()),
            "min": float(amounts.min()),
            "max": float(amounts.max())
        },
        "percentiles": percentiles,
        "distribution": distribution
    }

@mcp.tool(name="merchant_summary_all", description="Merchant summary for ALL merchants.")
def merchant_summary_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    merchant_data = df.groupby("merchant_category").agg({
        "transaction_id": "count",
        "amount": ["sum", "mean"],
        "is_fraud": "mean" if "is_fraud" in df.columns else "count"
    })
    merchant_data.columns = ['count', 'total', 'avg', 'fraud_rate']
    merchant_data = merchant_data.sort_values('total', ascending=False)
    
    merchants = {}
    for cat, row in merchant_data.iterrows():
        merchants[cat] = {
            "transactions": int(row['count']),
            "total_revenue": float(row['total']),
            "avg_transaction": float(row['avg']),
            "fraud_rate": float(row.get('fraud_rate', 0))
        }
    
    return {"merchants": merchants, "total_merchant_categories": len(merchants)}

@mcp.tool(name="top_merchants_all", description="Top merchants by spending for ALL users.")
def top_merchants_all(limit: int = Field(default=10)):
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    top = df.groupby("merchant_category")["amount"].agg(['sum', 'mean', 'count'])
    top = top.sort_values('sum', ascending=False).head(limit)
    
    return {
        "merchants": {k: {"total": float(v['sum']), "avg": float(v['mean']), "count": int(v['count'])} 
                      for k, v in top.iterrows()}
    }

@mcp.tool(name="card_usage_all", description="Card usage patterns for ALL users.")
def card_usage_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    chip_usage = df.groupby("use_chip")["amount"].sum()
    card_types = df.groupby("card_type")["amount"].agg(['sum', 'mean', 'count'])
    card_brands = df.groupby("card_brand")["amount"].agg(['sum', 'mean', 'count'])
    
    return {
        "chip_vs_swipe": {str(k): float(v) for k, v in chip_usage.items()},
        "by_card_type": {k: {"total": float(v['sum']), "avg": float(v['mean']), "count": int(v['count'])} 
                         for k, v in card_types.iterrows()},
        "by_card_brand": {k: {"total": float(v['sum']), "avg": float(v['mean']), "count": int(v['count'])} 
                         for k, v in card_brands.iterrows()}
    }

@mcp.tool(name="customer_segmentation_all", description="Customer segmentation for ALL users.")
def customer_segmentation_all():
    df = transactions
    if df.empty:
        return error_response("No data available")
    
    client_stats = df.groupby("client_id").agg({
        "amount": ["sum", "mean", "count"],
        "merchant_category": "nunique"
    })
    client_stats.columns = ['total_spent', 'avg_transaction', 'transaction_count', 'unique_merchants']
    
    segments = {
        "high_spenders": int((client_stats['total_spent'] > client_stats['total_spent'].quantile(0.75)).sum()),
        "loyal_customers": int((client_stats['transaction_count'] > client_stats['transaction_count'].quantile(0.75)).sum()),
        "diverse_shoppers": int((client_stats['unique_merchants'] > 10).sum()),
        "occasional_shoppers": int((client_stats['transaction_count'] <= 5).sum())
    }
    
    return {
        "segments": segments,
        "total_clients": len(client_stats),
        "avg_spent_per_client": float(client_stats['total_spent'].mean()),
        "avg_transactions_per_client": float(client_stats['transaction_count'].mean())
    }

@mcp.tool(name="spending_forecast_all", description="Spending forecast for ALL users.")
def spending_forecast_all(months: int = Field(default=3)):
    df = transactions.copy()
    if df.empty:
        return error_response("No data available")
    
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()
    
    avg_monthly = float(monthly.mean())
    trend = "increasing" if monthly.diff().mean() > 0 else "decreasing"
    
    forecast = {f"month_{i+1}": round(avg_monthly, 2) for i in range(months)}
    
    return {
        "forecast": forecast,
        "avg_historical_monthly": round(avg_monthly, 2),
        "trend": trend,
        "total_months_data": len(monthly)
    }

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
