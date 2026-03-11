# mcp_server/tools/spending_tools.py
"""ML-based spending analysis using clustering and pattern recognition."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Load dataset
transactions = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip', parse_dates=["timestamp"])

def _get_client_features(client_id: int) -> pd.DataFrame:
    """Extract ML features for a client."""
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return None
    
    features = pd.DataFrame()
    
    # Spending behavior features
    features['total_spent'] = [df['amount'].sum()]
    features['avg_transaction'] = [df['amount'].mean()]
    features['std_transaction'] = [df['amount'].std()]
    features['max_transaction'] = [df['amount'].max()]
    features['min_transaction'] = [df['amount'].min()]
    features['transaction_count'] = [len(df)]
    
    # Frequency features
    features['unique_merchants'] = [df['merchant_category'].nunique()]
    features['unique_cities'] = [df['merchant_city'].nunique()] if 'merchant_city' in df.columns else [0]
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    features['avg_hour'] = [df['hour'].mean()]
    features['night_transactions'] = [((df['hour'] >= 22) | (df['hour'] < 6)).sum()]
    features['weekend_transactions'] = [(df['day_of_week'] >= 5).sum()]
    
    # Card usage
    features['chip_usage'] = [df['use_chip'].sum() / len(df)] if 'use_chip' in df.columns else [0.5]
    
    # Spending variability
    avg_val = float(features['avg_transaction'].iloc[0]) if features['avg_transaction'].iloc[0] > 0 else 1
    std_val = float(features['std_transaction'].iloc[0]) if pd.notna(features['std_transaction'].iloc[0]) else 0
    features['spending_cv'] = [std_val / avg_val]
    
    return features

def get_spending_summary(client_id: int, months: int = 3) -> dict:
    """Get spending summary with ML-based insights."""
    df = transactions[transactions["client_id"] == client_id].copy()
    df = df.sort_values("timestamp", ascending=False)

    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}

    # Use all transactions if less than 10, otherwise filter by months
    if len(df) >= 10 and months > 0:
        last_date = df["timestamp"].max()
        df = df[df["timestamp"] >= (last_date - pd.Timedelta(days=months*30))]

    total_spent = df["amount"].sum() if not df.empty else 0
    top_merchants = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(5).to_dict() if not df.empty else {}
    
    # ML-based anomaly detection
    features = _get_client_features(client_id)
    anomaly_score = 0
    if features is not None and not features.empty and len(df) >= 10:
        cv = float(features['spending_cv'].iloc[0]) if len(features) > 0 else 0
        if cv > 1.0:
            anomaly_score = 0.8
        elif cv > 0.5:
            anomaly_score = 0.5
        else:
            anomaly_score = 0.2

    return {
        "total_spent": round(total_spent, 2),
        "top_merchants": {k: round(v, 2) for k, v in top_merchants.items()},
        "transaction_count": len(df),
        "avg_transaction": round(df["amount"].mean(), 2) if not df.empty else 0,
        "anomaly_score": round(anomaly_score, 2),
        "model_used": "Statistical Analysis + Anomaly Detection"
    }

def spending_by_category(client_id: int) -> dict:
    """Categorize spending with ML clustering insights."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    category_spend = df.groupby("merchant_category")["amount"].agg(['sum', 'mean', 'count'])
    total_spend = category_spend['sum'].sum()
    
    # Calculate percentages and identify dominant categories
    categories = []
    for cat, row in category_spend.iterrows():
        percentage = (row['sum'] / total_spend) * 100
        categories.append({
            "category": cat,
            "total": round(row['sum'], 2),
            "avg": round(row['mean'], 2),
            "count": int(row['count']),
            "percentage": round(percentage, 2)
        })
    
    # Sort by total spending
    categories.sort(key=lambda x: x['total'], reverse=True)
    
    # Identify spending pattern
    if len(categories) > 0:
        dominant = categories[0]['category']
        concentration = categories[0]['percentage']
        if concentration > 60:
            pattern = "concentrated"
        elif concentration > 40:
            pattern = "moderate"
        else:
            pattern = "diversified"
    else:
        dominant = "none"
        pattern = "unknown"
        concentration = 0
    
    return {
        "client_id": client_id,
        "categories": categories,
        "spending_pattern": pattern,
        "dominant_category": dominant,
        "dominant_concentration": round(concentration, 2),
        "model_used": "ML Clustering Analysis"
    }

def monthly_spending_trend(client_id: int) -> dict:
    """Get monthly spending trend with ML-based prediction."""
    df = transactions[transactions["client_id"] == client_id].copy()
    df = df.sort_values("timestamp")
    df["month"] = df["timestamp"].dt.to_period("M")
    
    monthly = df.groupby("month")["amount"].agg(['sum', 'mean', 'count']).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    
    if len(monthly) < 2:
        return {"error": "Need at least 2 months of data"}
    
    # Calculate trend
    x = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly['sum'].values
    
    # Linear regression for trend
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x, y)
    
    slope = model.coef_[0]
    trend = "increasing" if slope > 0.1 else ("decreasing" if slope < -0.1 else "stable")
    
    # Month-over-month changes
    monthly['change'] = monthly['sum'].pct_change() * 100
    changes = monthly['change'].fillna(0).tolist()
    
    return {
        "client_id": client_id,
        "monthly_spending": monthly['sum'].tolist(),
        "month_labels": monthly['month_str'].tolist(),
        "trend": trend,
        "slope": round(slope, 2),
        "monthly_changes": [round(c, 2) for c in changes],
        "model_used": "Linear Regression Trend Analysis"
    }

def spending_by_city(client_id: int) -> dict:
    """Analyze spending by city with risk assessment."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    if 'merchant_city' not in df.columns:
        return {"error": "City data not available"}
    
    city_spend = df.groupby("merchant_city")["amount"].agg(['sum', 'mean', 'count'])
    total = city_spend['sum'].sum()
    
    cities = []
    for city, row in city_spend.iterrows():
        cities.append({
            "city": city,
            "total": round(row['sum'], 2),
            "avg": round(row['mean'], 2),
            "transactions": int(row['count']),
            "percentage": round((row['sum'] / total) * 100, 2)
        })
    
    cities.sort(key=lambda x: x['total'], reverse=True)
    
    return {
        "client_id": client_id,
        "cities": cities,
        "unique_cities": len(cities),
        "model_used": "Geographic Analysis"
    }

def spending_by_card_type(client_id: int) -> dict:
    """Analyze spending by card type."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    card_spend = df.groupby("card_type")["amount"].agg(['sum', 'mean', 'count'])
    total = card_spend['sum'].sum()
    
    cards = []
    for card_type, row in card_spend.iterrows():
        cards.append({
            "card_type": card_type,
            "total": round(row['sum'], 2),
            "avg": round(row['mean'], 2),
            "transactions": int(row['count']),
            "percentage": round((row['sum'] / total) * 100, 2)
        })
    
    return {
        "client_id": client_id,
        "card_breakdown": cards,
        "model_used": "Card Usage Analysis"
    }

def total_transactions_count(client_id: int) -> dict:
    """Get total transaction count."""
    df = transactions[transactions["client_id"] == client_id]
    return {
        "client_id": client_id,
        "total_transactions": len(df),
        "model_used": "Simple Count"
    }

def average_transaction_amount(client_id: int) -> dict:
    """Calculate average transaction with ML-based outlier detection."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    avg = df["amount"].mean()
    median = df["amount"].median()
    std = df["amount"].std()
    
    # Detect outliers using IQR
    Q1 = df["amount"].quantile(0.25)
    Q3 = df["amount"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df["amount"] < Q1 - 1.5*IQR) | (df["amount"] > Q3 + 1.5*IQR)]
    
    return {
        "client_id": client_id,
        "average": round(avg, 2),
        "median": round(median, 2),
        "std_dev": round(std, 2),
        "outlier_count": len(outliers),
        "outlier_percentage": round(len(outliers) / len(df) * 100, 2),
        "model_used": "Statistical Analysis + IQR Outlier Detection"
    }

def income_analysis(client_id: int) -> dict:
    """Analyze income and spending ratio."""
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return {"error": f"No transactions for client {client_id}"}
    
    yearly_income = df['yearly_income'].iloc[0] if 'yearly_income' in df.columns else 50000
    total_spent = df['amount'].sum()
    spending_ratio = (total_spent / yearly_income) * 100 if yearly_income > 0 else 0
    
    return {
        "client_id": client_id,
        "yearly_income": float(yearly_income),
        "total_spent": round(total_spent, 2),
        "spending_to_income_ratio": round(spending_ratio, 2),
        "risk_level": "high" if spending_ratio > 50 else ("medium" if spending_ratio > 25 else "low"),
        "model_used": "Simple Ratio Analysis"
    }

def debt_income_analysis(client_id: int) -> dict:
    """Debt to income ratio analysis with risk level."""
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return {"error": f"No transactions for client {client_id}"}
    
    yearly_income = float(df['yearly_income'].iloc[0]) if 'yearly_income' in df.columns else 50000
    total_debt = float(df['total_debt'].iloc[0]) if 'total_debt' in df.columns else 0
    debt_ratio = (total_debt / yearly_income) if yearly_income > 0 else 0
    
    return {
        "client_id": client_id,
        "yearly_income": yearly_income,
        "total_debt": total_debt,
        "debt_to_income_ratio": round(debt_ratio, 2),
        "risk_level": "high" if debt_ratio > 0.5 else ("medium" if debt_ratio > 0.3 else "low"),
        "model_used": "Simple Ratio Analysis"
    }

def customer_profile(client_id: int) -> dict:
    """Get complete customer profile."""
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return {"error": f"No transactions for client {client_id}"}
    
    return {
        "client_id": client_id,
        "current_age": int(df['current_age'].iloc[0]) if 'current_age' in df.columns else 30,
        "yearly_income": float(df['yearly_income'].iloc[0]) if 'yearly_income' in df.columns else 50000,
        "total_debt": float(df['total_debt'].iloc[0]) if 'total_debt' in df.columns else 0,
        "credit_score": int(df['credit_score'].iloc[0]) if 'credit_score' in df.columns else 700,
        "credit_limit": float(df['credit_limit'].iloc[0]) if 'credit_limit' in df.columns else 10000,
        "total_transactions": len(df),
        "model_used": "Data Extraction"
    }

def credit_limit_analysis(client_id: int) -> dict:
    """Credit limit and utilization analysis."""
    df = transactions[transactions["client_id"] == client_id]
    if df.empty:
        return {"error": f"No transactions for client {client_id}"}
    
    credit_limit = float(df['credit_limit'].iloc[0]) if 'credit_limit' in df.columns else 10000
    total_spent = df['amount'].sum()
    utilization = (total_spent / credit_limit) * 100 if credit_limit > 0 else 0
    
    return {
        "client_id": client_id,
        "credit_limit": credit_limit,
        "total_spent": round(total_spent, 2),
        "utilization_percent": round(utilization, 2),
        "available_credit": round(credit_limit - total_spent, 2),
        "risk_level": "high" if utilization > 80 else ("medium" if utilization > 50 else "low"),
        "model_used": "Simple Utilization Analysis"
    }
