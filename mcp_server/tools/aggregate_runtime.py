"""Aggregate all-user MCP tools."""

import pandas as pd
from pydantic import Field

from .common import error_response, get_cached, get_transactions


def register_tools(mcp):
    @mcp.tool(name="get_all_users_spending_summary", description="Get spending summary for ALL users/clients combined.")
    def get_all_users_spending_summary(months: int = Field(default=3)):
        df = get_transactions().copy()
        if df.empty:
            return error_response("No transaction data available")

        last_date = df["timestamp"].max()
        df = df[df["timestamp"] >= (last_date - pd.Timedelta(days=months * 30))]
        total = float(df["amount"].sum())
        unique_clients = df["client_id"].nunique()
        transaction_count = len(df)
        avg_per_transaction = float(df["amount"].mean())
        avg_per_client = total / unique_clients if unique_clients > 0 else 0
        top_merchants = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=False).head(5)

        client_breakdown = df.groupby("client_id").agg({
            "amount": ["sum", "mean", "count"],
            "yearly_income": "first",
            "credit_score": "first",
            "current_age": "first",
        }).reset_index()
        client_breakdown.columns = ["client_id", "total_spent", "avg_transaction", "transaction_count", "yearly_income", "credit_score", "age"]

        client_name_map = {
            1: "Bob", 2: "Charlie", 3: "David", 4: "Soniya", 5: "Frank",
            6: "Grace", 7: "Sarah", 8: "Henry", 9: "Ivy", 10: "Jack",
            11: "John", 12: "Kate", 13: "Leo", 14: "Emma", 15: "Nathan",
            16: "Olivia", 17: "Peter", 18: "Client_18", 19: "Client_19", 20: "Mike",
            21: "Ava", 22: "Liam", 23: "Noah", 24: "Mia", 25: "Zoe",
            26: "Ethan", 27: "Nora", 28: "Lucas", 29: "Isla", 30: "Owen",
        }

        client_details = []
        for _, row in client_breakdown.iterrows():
            cid = int(row["client_id"])
            client_details.append({
                "client_id": cid,
                "name": client_name_map.get(cid, f"Client_{cid}"),
                "total_spent": round(float(row["total_spent"]), 2),
                "avg_transaction": round(float(row["avg_transaction"]), 2),
                "transaction_count": int(row["transaction_count"]),
                "yearly_income": float(row["yearly_income"]) if pd.notna(row["yearly_income"]) else 0,
                "credit_score": int(row["credit_score"]) if pd.notna(row["credit_score"]) else 0,
                "age": int(row["age"]) if pd.notna(row["age"]) else 0,
            })
        client_details.sort(key=lambda x: x["total_spent"], reverse=True)

        return {
            "total_spent": total,
            "unique_clients": int(unique_clients),
            "transaction_count": transaction_count,
            "avg_per_transaction": round(avg_per_transaction, 2),
            "avg_per_client": round(avg_per_client, 2),
            "top_merchants": {k: float(v) for k, v in top_merchants.items()},
            "months": months,
            "client_details": client_details,
        }

    @mcp.tool(name="spending_by_category_all", description="Spending breakdown by merchant category for ALL users.")
    def spending_by_category_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        cat_spend = df.groupby("merchant_category")["amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        categories = {cat: {"total": float(row["sum"]), "avg": float(row["mean"]), "count": int(row["count"])} for cat, row in cat_spend.iterrows()}
        return {"categories": categories, "total_unique_categories": len(categories)}

    @mcp.tool(name="monthly_spending_trend_all", description="Monthly spending trend for ALL users.")
    def monthly_spending_trend_all():
        df = get_transactions().copy()
        if df.empty:
            return error_response("No data available")
        df["month"] = df["timestamp"].dt.to_period("M")
        monthly = df.groupby("month")["amount"].agg(["sum", "mean", "count"]).sort_index()
        return {"months": {str(k): {"total": float(v["sum"]), "avg": float(v["mean"]), "count": int(v["count"])} for k, v in monthly.iterrows()}}

    @mcp.tool(name="spending_by_city_all", description="Spending breakdown by city for ALL users.")
    def spending_by_city_all():
        df = get_transactions()
        if df.empty or "merchant_city" not in df.columns:
            return error_response("No city data available")
        city_spend = df.groupby("merchant_city")["amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False).head(20)
        cities = {city: {"total": float(row["sum"]), "avg": float(row["mean"]), "count": int(row["count"])} for city, row in city_spend.iterrows()}
        return {"cities": cities, "total_unique_cities": len(cities)}

    @mcp.tool(name="spending_by_card_type_all", description="Spending by card type for ALL users.")
    def spending_by_card_type_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        card_spend = df.groupby("card_type")["amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        cards = {card: {"total": float(row["sum"]), "avg": float(row["mean"]), "count": int(row["count"])} for card, row in card_spend.iterrows()}
        return {"card_types": cards}

    @mcp.tool(name="total_transactions_all", description="Total transactions count for ALL users.")
    def total_transactions_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        return {"total_transactions": len(df), "unique_clients": int(df["client_id"].nunique()), "unique_merchants": int(df["merchant_category"].nunique())}

    @mcp.tool(name="average_transaction_all", description="Average transaction amount for ALL users.")
    def average_transaction_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        amounts = df["amount"]
        return {"average": float(amounts.mean()), "median": float(amounts.median()), "min": float(amounts.min()), "max": float(amounts.max()), "std": float(amounts.std()), "total_transactions": len(df)}

    @mcp.tool(name="global_fraud_analysis", description="Global fraud analysis for ALL transactions.")
    def global_fraud_analysis():
        return get_cached("global_fraud_analysis", _compute_global_fraud)

    def _compute_global_fraud():
        df = get_transactions()
        if df.empty or "is_fraud" not in df.columns:
            return error_response("No fraud data available")
        total = len(df)
        fraud_count = int(df["is_fraud"].sum())
        fraud_rate = fraud_count / total if total > 0 else 0
        fraud_by_category = df.groupby("merchant_category")["is_fraud"].agg(["sum", "mean"])
        fraud_by_card = df.groupby("card_type")["is_fraud"].agg(["sum", "mean"])
        return {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_rate, 4),
            "fraud_by_category": {cat: {"count": int(row["sum"]), "rate": float(row["mean"])} for cat, row in fraud_by_category.iterrows() if row["sum"] > 0},
            "fraud_by_card_type": {cat: {"count": int(row["sum"]), "rate": float(row["mean"])} for cat, row in fraud_by_card.iterrows() if row["sum"] > 0},
        }

    @mcp.tool(name="all_clients_fraud_summary", description="Fraud summary for ALL clients.")
    def all_clients_fraud_summary():
        return get_cached("all_clients_fraud_summary", _compute_all_clients_fraud)

    def _compute_all_clients_fraud():
        df = get_transactions()
        if df.empty or "is_fraud" not in df.columns:
            return error_response("No fraud data available")
        client_fraud = df.groupby("client_id").agg({"transaction_id": "count", "amount": "sum", "is_fraud": "sum"}).reset_index()
        client_fraud["fraud_rate"] = client_fraud["is_fraud"] / client_fraud["transaction_id"]
        top_fraudulent = client_fraud.nlargest(10, "fraud_rate")
        return {
            "total_clients": int(df["client_id"].nunique()),
            "clients_with_fraud": int((client_fraud["is_fraud"] > 0).sum()),
            "top_fraudulent_clients": [{"client_id": int(row["client_id"]), "fraud_rate": round(float(row["fraud_rate"]), 4), "fraud_count": int(row["is_fraud"])} for _, row in top_fraudulent.iterrows()],
        }

    @mcp.tool(name="credit_score_distribution_all", description="Credit score distribution for ALL users.")
    def credit_score_distribution_all():
        df = get_transactions()
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
                "poor_below_650": int((scores < 650).sum()),
            },
            "total_clients": len(scores),
        }

    @mcp.tool(name="debt_income_all", description="Debt to income analysis for ALL users.")
    def debt_income_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        client_data = df.groupby("client_id").agg({"yearly_income": "first", "total_debt": "first"}).reset_index()
        client_data["debt_to_income"] = client_data["total_debt"] / client_data["yearly_income"].replace(0, 1)
        return {"mean_debt_to_income": round(float(client_data["debt_to_income"].mean()), 3), "median_debt_to_income": round(float(client_data["debt_to_income"].median()), 3), "high_risk_clients": int((client_data["debt_to_income"] > 0.5).sum()), "medium_risk_clients": int(((client_data["debt_to_income"] > 0.3) & (client_data["debt_to_income"] <= 0.5)).sum()), "low_risk_clients": int((client_data["debt_to_income"] <= 0.3).sum()), "total_clients": len(client_data)}

    @mcp.tool(name="credit_limit_all", description="Credit limit and utilization for ALL users.")
    def credit_limit_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        client_data = df.groupby("client_id").agg({"credit_limit": "first", "amount": "sum"}).reset_index()
        client_data["utilization"] = (client_data["amount"] / client_data["credit_limit"].replace(0, 1)) * 100
        return {"mean_credit_limit": float(client_data["credit_limit"].mean()), "mean_utilization": round(float(client_data["utilization"].mean()), 2), "high_utilization_clients": int((client_data["utilization"] > 75).sum()), "over_limit_clients": int((client_data["utilization"] > 100).sum()), "total_clients": len(client_data)}

    @mcp.tool(name="transaction_heatmap_all", description="Transaction heatmap for ALL users.")
    def transaction_heatmap_all():
        df = get_transactions().copy()
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
        return {"heatmap": result, "peak_day": day_names[int(peak_day_hour[0])], "peak_hour": int(peak_day_hour[1]), "total_transactions": len(df)}

    @mcp.tool(name="peak_hours_all", description="Peak transaction hours for ALL users.")
    def peak_hours_all():
        df = get_transactions().copy()
        if df.empty:
            return error_response("No data available")
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        hourly = df.groupby("hour")["amount"].agg(["sum", "mean", "count"])
        daily = df.groupby("day_of_week")["amount"].agg(["sum", "mean", "count"])
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return {"by_hour": {int(k): float(v["sum"]) for k, v in hourly.iterrows()}, "by_day": {day_names[int(k)]: float(v["sum"]) for k, v in daily.iterrows()}, "peak_hour": int(hourly["sum"].idxmax()), "peak_day": day_names[int(daily["sum"].idxmax())]}

    @mcp.tool(name="spending_distribution_all", description="Spending distribution for ALL users.")
    def spending_distribution_all():
        df = get_transactions().copy()
        if df.empty:
            return error_response("No data available")
        amounts = df["amount"]
        percentiles = {"25%": float(amounts.quantile(0.25)), "50%": float(amounts.quantile(0.50)), "75%": float(amounts.quantile(0.75)), "90%": float(amounts.quantile(0.90)), "95%": float(amounts.quantile(0.95)), "99%": float(amounts.quantile(0.99))}
        bins = [0, 20, 50, 100, 200, 500, 1000, float("inf")]
        labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
        df["bin"] = pd.cut(df["amount"], bins=bins, labels=labels)
        return {"statistics": {"mean": float(amounts.mean()), "median": float(amounts.median()), "std": float(amounts.std()), "min": float(amounts.min()), "max": float(amounts.max())}, "percentiles": percentiles, "distribution": df["bin"].value_counts().sort_index().to_dict()}

    @mcp.tool(name="merchant_summary_all", description="Merchant summary for ALL merchants.")
    def merchant_summary_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        merchant_data = df.groupby("merchant_category").agg({"transaction_id": "count", "amount": ["sum", "mean"], "is_fraud": "mean" if "is_fraud" in df.columns else "count"})
        merchant_data.columns = ["count", "total", "avg", "fraud_rate"]
        merchant_data = merchant_data.sort_values("total", ascending=False)
        merchants = {}
        for cat, row in merchant_data.iterrows():
            merchants[cat] = {"transactions": int(row["count"]), "total_revenue": float(row["total"]), "avg_transaction": float(row["avg"]), "fraud_rate": float(row.get("fraud_rate", 0))}
        return {"merchants": merchants, "total_merchant_categories": len(merchants)}

    @mcp.tool(name="top_merchants_all", description="Top merchants by spending for ALL users.")
    def top_merchants_all(limit: int = Field(default=10)):
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        top = df.groupby("merchant_category")["amount"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False).head(limit)
        return {"merchants": {k: {"total": float(v["sum"]), "avg": float(v["mean"]), "count": int(v["count"])} for k, v in top.iterrows()}}

    @mcp.tool(name="card_usage_all", description="Card usage patterns for ALL users.")
    def card_usage_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        chip_usage = df.groupby("use_chip")["amount"].sum()
        card_types = df.groupby("card_type")["amount"].agg(["sum", "mean", "count"])
        card_brands = df.groupby("card_brand")["amount"].agg(["sum", "mean", "count"])
        return {"chip_vs_swipe": {str(k): float(v) for k, v in chip_usage.items()}, "by_card_type": {k: {"total": float(v["sum"]), "avg": float(v["mean"]), "count": int(v["count"])} for k, v in card_types.iterrows()}, "by_card_brand": {k: {"total": float(v["sum"]), "avg": float(v["mean"]), "count": int(v["count"])} for k, v in card_brands.iterrows()}}

    @mcp.tool(name="customer_segmentation_all", description="Customer segmentation for ALL users.")
    def customer_segmentation_all():
        df = get_transactions()
        if df.empty:
            return error_response("No data available")
        client_stats = df.groupby("client_id").agg({"amount": ["sum", "mean", "count"], "merchant_category": "nunique"})
        client_stats.columns = ["total_spent", "avg_transaction", "transaction_count", "unique_merchants"]
        segments = {"high_spenders": int((client_stats["total_spent"] > client_stats["total_spent"].quantile(0.75)).sum()), "loyal_customers": int((client_stats["transaction_count"] > client_stats["transaction_count"].quantile(0.75)).sum()), "diverse_shoppers": int((client_stats["unique_merchants"] > 10).sum()), "occasional_shoppers": int((client_stats["transaction_count"] <= 5).sum())}
        return {"segments": segments, "total_clients": len(client_stats), "avg_spent_per_client": float(client_stats["total_spent"].mean()), "avg_transactions_per_client": float(client_stats["transaction_count"].mean())}

    @mcp.tool(name="spending_forecast_all", description="Spending forecast for ALL users.")
    def spending_forecast_all(months: int = Field(default=3)):
        df = get_transactions().copy()
        if df.empty:
            return error_response("No data available")
        df["month"] = df["timestamp"].dt.to_period("M")
        monthly = df.groupby("month")["amount"].sum()
        avg_monthly = float(monthly.mean())
        trend = "increasing" if monthly.diff().mean() > 0 else "decreasing"
        forecast = {f"month_{i + 1}": round(avg_monthly, 2) for i in range(months)}
        return {"forecast": forecast, "avg_historical_monthly": round(avg_monthly, 2), "trend": trend, "total_months_data": len(monthly)}
