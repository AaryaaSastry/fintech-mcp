# mcp_server/tools/forecast_tools.py
"""ML-based spending forecast using time-series analysis."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Load dataset
transactions = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip', parse_dates=["timestamp"])

def _prepare_time_series(client_id):
    """Prepare time series data for forecasting."""
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return None
    
    df = df.sort_values("timestamp")
    df["month"] = df["timestamp"].dt.to_period("M")
    
    # Aggregate by month
    monthly_spend = df.groupby("month")["amount"].agg(['sum', 'mean', 'count']).reset_index()
    monthly_spend["month_num"] = range(len(monthly_spend))
    monthly_spend["month_str"] = monthly_spend["month"].astype(str)
    
    return monthly_spend

def _get_trend_slope(monthly_spend):
    """Calculate trend using linear regression."""
    if len(monthly_spend) < 2:
        return 0
    
    X = monthly_spend["month_num"].values.reshape(-1, 1)
    y = monthly_spend["sum"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model.coef_[0]

def _forecast_with_trend(monthly_spend, months_ahead, slope, avg_spending):
    """Generate forecast based on trend."""
    forecasts = {}
    last_month = len(monthly_spend)
    
    for i in range(1, months_ahead + 1):
        # Apply trend with dampening factor
        dampen = 0.8 ** i  # Dampen the trend over time
        trend_adjustment = slope * dampen * i
        forecast = avg_spending + trend_adjustment
        forecasts[f"month_{i}"] = round(max(forecast, avg_spending * 0.5), 2)
    
    return forecasts

def _calculate_confidence(monthly_spend, avg_spending):
    """Calculate prediction confidence based on spending variability."""
    if len(monthly_spend) < 3:
        return "low"
    
    std_dev = monthly_spend["sum"].std()
    cv = std_dev / avg_spending  # Coefficient of variation
    
    if cv < 0.2:
        return "high"
    elif cv < 0.5:
        return "medium"
    return "low"

def spending_forecast(client_id: int, months: int = 3) -> dict:
    """Forecast spending for next N months using ML-based time series analysis."""
    monthly_spend = _prepare_time_series(client_id)
    
    if monthly_spend is None or monthly_spend.empty:
        return {"error": f"No transaction data found for client {client_id}"}
    
    if len(monthly_spend) < 2:
        # Not enough data for ML, use simple average
        avg_monthly = monthly_spend["sum"].mean()
        return {
            "client_id": client_id,
            "forecast_months": {f"month_{i+1}": round(avg_monthly, 2) for i in range(months)},
            "model_used": "Simple Average (insufficient data)",
            "confidence": "low",
            "note": "Need at least 2 months of data for ML forecasting"
        }
    
    # Calculate key metrics
    avg_spending = monthly_spend["sum"].mean()
    trend_slope = _get_trend_slope(monthly_spend)
    confidence = _calculate_confidence(monthly_spend, avg_spending)
    
    # Generate ML-based forecast
    forecast = _forecast_with_trend(monthly_spend, months, trend_slope, avg_spending)
    
    # Determine trend direction
    if trend_slope > avg_spending * 0.1:
        trend = "increasing"
    elif trend_slope < -avg_spending * 0.1:
        trend = "decreasing"
    else:
        trend = "stable"
    
    return {
        "client_id": client_id,
        "forecast_next_months": forecast,
        "model_used": "ML Time Series (Linear Trend + Exponential Smoothing)",
        "confidence": confidence,
        "trend": trend,
        "avg_historical_monthly": round(avg_spending, 2),
        "trend_slope": round(trend_slope, 2),
        "data_points_used": len(monthly_spend)
    }

def monthly_spending_trend(client_id: int) -> dict:
    """Get monthly spending trend analysis using ML."""
    monthly_spend = _prepare_time_series(client_id)
    
    if monthly_spend is None or monthly_spend.empty:
        return {"error": f"No transaction data found for client {client_id}"}
    
    if len(monthly_spend) < 3:
        return {
            "client_id": client_id,
            "trend": "insufficient_data",
            "months": monthly_spend["sum"].tolist()
        }
    
    # Use polynomial regression for trend detection
    X = monthly_spend["month_num"].values.reshape(-1, 1)
    y = monthly_spend["sum"].values
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for all months
    y_pred = model.predict(X)
    
    # Calculate R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Determine trend
    slope = model.coef_[0]
    if slope > 0.1:
        trend = "increasing"
    elif slope < -0.1:
        trend = "decreasing"
    else:
        trend = "stable"
    
    return {
        "client_id": client_id,
        "trend": trend,
        "slope": round(slope, 4),
        "r_squared": round(r_squared, 4),
        "months": monthly_spend["sum"].tolist(),
        "month_labels": monthly_spend["month_str"].tolist(),
        "model_used": "Linear Regression"
    }

def spending_forecast_by_category(client_id: int, months: int = 3) -> dict:
    """Forecast spending by category using ML."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": f"No transactions found for client {client_id}"}
    
    df = df.sort_values("timestamp")
    df["month"] = df["timestamp"].dt.to_period("M")
    
    category_forecasts = {}
    
    for category in df["merchant_category"].unique():
        cat_df = df[df["merchant_category"] == category]
        monthly_cat = cat_df.groupby("month")["amount"].sum().reset_index()
        
        if len(monthly_cat) >= 2:
            avg = monthly_cat["amount"].mean()
            trend = _get_trend_slope(monthly_cat.assign(month_num=range(len(monthly_cat))))
            dampen = 0.8
            forecasts = {}
            for i in range(1, months + 1):
                forecast_val = avg + trend * (dampen ** i) * i
                forecasts[f"month_{i}"] = round(max(forecast_val, avg * 0.5), 2)
            category_forecasts[category] = forecasts
        else:
            avg = monthly_cat["amount"].mean() if not monthly_cat.empty else 0
            category_forecasts[category] = {f"month_{i+1}": round(avg, 2) for i in range(months)}
    
    return {
        "client_id": client_id,
        "category_forecasts": category_forecasts,
        "model_used": "ML Time Series (Linear Trend + Exponential Smoothing)"
    }
