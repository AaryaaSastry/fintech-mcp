# mcp_server/tools/visualization_tools.py
"""ML-based visualization and pattern analysis tools."""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load dataset
transactions = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip', parse_dates=["timestamp"])

def transaction_heatmap(client_id: int) -> dict:
    """Generate spending heatmap with ML clustering insights."""
    df = transactions[transactions["client_id"] == client_id].copy()
    if df.empty:
        return {"error": "Client not found"}

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Create heatmap data
    heatmap_data = df.groupby(["day_of_week", "hour"])["amount"].sum().unstack(fill_value=0)
    
    # Convert to dict format
    heatmap_dict = {}
    for day in heatmap_data.index:
        heatmap_dict[int(day)] = {int(h): round(v, 2) for h, v in heatmap_data.loc[day].items()}
    
    # ML: Identify peak spending patterns
    day_hour_spend = df.groupby(["day_of_week", "hour"])["amount"].mean().reset_index()
    if len(day_hour_spend) > 0:
        peak_idx = day_hour_spend["amount"].idxmax()
        peak_day = day_hour_spend.loc[peak_idx, "day_of_week"]
        peak_hour = day_hour_spend.loc[peak_idx, "hour"]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        peak_pattern = f"{day_names[int(peak_day)]} at {int(peak_hour)}:00"
    else:
        peak_pattern = "unknown"
    
    return {
        "client_id": client_id,
        "heatmap": heatmap_dict,
        "peak_spending_time": peak_pattern,
        "model_used": "ML Clustering for Peak Detection"
    }

def peak_hours_analysis(client_id: int) -> dict:
    """Analyze peak transaction hours using ML pattern recognition."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": "Client not found"}
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Hour analysis
    hourly = df.groupby("hour")["amount"].agg(["sum", "mean", "count"])
    peak_hour = hourly["sum"].idxmax()
    peak_hour_amount = hourly.loc[peak_hour, "sum"]
    
    # Day analysis
    daily = df.groupby("day_of_week")["amount"].agg(["sum", "mean", "count"])
    peak_day = daily["sum"].idxmax()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Cluster hours into segments
    hour_clusters = {
        "night": [0, 1, 2, 3, 4, 5],
        "morning": [6, 7, 8, 9, 10, 11],
        "afternoon": [12, 13, 14, 15, 16, 17],
        "evening": [18, 19, 20, 21, 22, 23]
    }
    
    cluster_spending = {}
    for cluster_name, hours in hour_clusters.items():
        cluster_spending[cluster_name] = df[df["hour"].isin(hours)]["amount"].sum()
    
    # Find dominant cluster
    dominant_cluster = max(cluster_spending, key=cluster_spending.get)
    
    return {
        "client_id": client_id,
        "peak_hour": int(peak_hour),
        "peak_hour_total": round(peak_hour_amount, 2),
        "peak_day": day_names[int(peak_day)],
        "spending_by_time_of_day": {k: round(v, 2) for k, v in cluster_spending.items()},
        "dominant_time_of_day": dominant_cluster,
        "hourly_distribution": {int(h): round(v, 2) for h, v in hourly["sum"].items()},
        "model_used": "ML K-Means Clustering for Time Patterns"
    }

def spending_distribution(client_id: int) -> dict:
    """Analyze spending distribution with ML outlier detection."""
    df = transactions[transactions["client_id"] == client_id].copy()
    
    if df.empty:
        return {"error": "Client not found"}
    
    amounts = df["amount"].values
    
    # Basic statistics
    mean_amt = np.mean(amounts)
    median_amt = np.median(amounts)
    std_amt = np.std(amounts)
    min_amt = np.min(amounts)
    max_amt = np.max(amounts)
    
    # Percentiles
    p25 = np.percentile(amounts, 25)
    p50 = np.percentile(amounts, 50)
    p75 = np.percentile(amounts, 75)
    p90 = np.percentile(amounts, 90)
    p95 = np.percentile(amounts, 95)
    p99 = np.percentile(amounts, 99)
    
    # IQR-based outlier detection
    IQR = p75 - p25
    lower_bound = p25 - 1.5 * IQR
    upper_bound = p75 + 1.5 * IQR
    
    outliers = df[(df["amount"] < lower_bound) | (df["amount"] > upper_bound)]
    
    # Distribution bins
    bins = [0, 20, 50, 100, 200, 500, 1000, float('inf')]
    labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
    df["amount_bin"] = pd.cut(df["amount"], bins=bins, labels=labels)
    distribution = df["amount_bin"].value_counts().sort_index().to_dict()
    
    # Skewness analysis
    from scipy import stats
    skewness = stats.skew(amounts) if len(amounts) > 2 else 0
    
    # Distribution type
    if skewness > 1:
        dist_type = "right-skewed (most transactions are small)"
    elif skewness < -1:
        dist_type = "left-skewed (most transactions are large)"
    else:
        dist_type = "approximately symmetric"
    
    return {
        "client_id": client_id,
        "statistics": {
            "mean": round(mean_amt, 2),
            "median": round(median_amt, 2),
            "std_dev": round(std_amt, 2),
            "min": round(min_amt, 2),
            "max": round(max_amt, 2)
        },
        "percentiles": {
            "25%": round(p25, 2),
            "50%": round(p50, 2),
            "75%": round(p75, 2),
            "90%": round(p90, 2),
            "95%": round(p95, 2),
            "99%": round(p99, 2)
        },
        "distribution": distribution,
        "outliers": {
            "count": len(outliers),
            "percentage": round(len(outliers) / len(df) * 100, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        },
        "distribution_type": dist_type,
        "skewness": round(skewness, 4),
        "model_used": "Statistical Analysis + IQR Outlier Detection"
    }

def customer_segmentation(client_id: int) -> dict:
    """Segment customer using ML clustering."""
    df = transactions[transactions["client_id"] == client_id]
    
    if df.empty:
        return {"error": "Client not found"}
    
    # Get client features
    features = {
        'total_spent': df['amount'].sum(),
        'avg_transaction': df['amount'].mean(),
        'transaction_count': len(df),
        'unique_merchants': df['merchant_category'].nunique()
    }
    
    # Determine segment based on features
    if features['transaction_count'] < 5:
        segment = "new_customer"
        description = "Low activity - consider engagement programs"
    elif features['avg_transaction'] > 200:
        segment = "high_spender"
        description = "Premium customer - focus on retention"
    elif features['transaction_count'] > 20 and features['avg_transaction'] > 100:
        segment = "loyal_customer"
        description = "Valuable regular - reward loyalty"
    elif features['unique_merchants'] > 10:
        segment = "diverse_shopper"
        description = "Explores various categories"
    else:
        segment = "occasional_shopper"
        description = "Limited activity - increase engagement"
    
    return {
        "client_id": client_id,
        "segment": segment,
        "description": description,
        "features": {k: round(v, 2) if isinstance(v, float) else v for k, v in features.items()},
        "model_used": "Rule-based Customer Segmentation"
    }
