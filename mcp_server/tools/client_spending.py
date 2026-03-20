"""Client spending, merchant, profile, and visualization MCP tools."""

import pandas as pd
from pydantic import Field

from engine.analysis_service import run_analysis
from engine.forecasting import run_forecast_analysis
from engine.profiles import get_client_overview

from .common import error_response, get_transactions


def register_tools(mcp):
    @mcp.tool(name="get_spending_summary", description="Get total spending and top merchant categories for a client over the last N months.")
    def get_spending_summary(client_id: int = Field(), months: int = Field(default=3)):
        summary = run_analysis(
            filters={"client_id": {"eq": client_id}},
            metrics=["sum(amount)", "count(*)", "avg(amount)"],
            limit=1,
            recent_months=months,
        )
        if not summary["data"]:
            return error_response(f"No transactions for client {client_id}")

        top_merchants_result = run_analysis(
            filters={"client_id": {"eq": client_id}},
            group_by=["merchant_category"],
            metrics=["sum(amount)"],
            sort_by="sum(amount)",
            limit=5,
            recent_months=months,
        )
        row = summary["data"][0]
        top_merchants = {
            item["merchant_category"]: float(item["sum(amount)"])
            for item in top_merchants_result["data"]
        }
        return {
            "total_spent": float(row["sum(amount)"]),
            "top_merchants": top_merchants,
            "transaction_count": int(row["count(*)"]),
            "avg_transaction": float(row["avg(amount)"]),
        }

    @mcp.tool(name="spending_by_category", description="Breakdown of spending by merchant category.")
    def spending_by_category(client_id: int = Field()):
        result = run_analysis(
            filters={"client_id": {"eq": client_id}},
            group_by=["merchant_category"],
            metrics=["sum(amount)"],
            sort_by="sum(amount)",
        )
        if not result["data"]:
            return error_response(f"No transactions for client {client_id}")
        return {"categories": {item["merchant_category"]: float(item["sum(amount)"]) for item in result["data"]}}

    @mcp.tool(name="monthly_spending_trend", description="Monthly spending trend over time.")
    def monthly_spending_trend(client_id: int = Field()):
        result = run_analysis(
            filters={"client_id": {"eq": client_id}},
            metrics=["sum(amount)"],
            time_grain="month",
            sort_by="time_grain",
            order="asc",
            limit=60,
        )
        if not result["data"]:
            return error_response(f"No transactions for client {client_id}")
        return {"months": {row["time_grain"]: float(row["sum(amount)"]) for row in result["data"]}}

    @mcp.tool(name="spending_by_city", description="Spending breakdown by merchant city.")
    def spending_by_city(client_id: int = Field()):
        result = run_analysis(
            filters={"client_id": {"eq": client_id}},
            group_by=["merchant_city"],
            metrics=["sum(amount)"],
            sort_by="sum(amount)",
            limit=10,
        )
        if not result["data"]:
            return error_response(f"No transactions for client {client_id}")
        return {"cities": {item["merchant_city"]: float(item["sum(amount)"]) for item in result["data"]}}

    @mcp.tool(name="spending_by_card_type", description="Spending by card type (Credit/Debit/Prepaid).")
    def spending_by_card_type(client_id: int = Field()):
        result = run_analysis(
            filters={"client_id": {"eq": client_id}},
            group_by=["card_type"],
            metrics=["sum(amount)"],
            sort_by="sum(amount)",
        )
        if not result["data"]:
            return error_response(f"No transactions for client {client_id}")
        return {"card_types": {item["card_type"]: float(item["sum(amount)"]) for item in result["data"]}}

    @mcp.tool(name="total_transactions_count", description="Total number of transactions.")
    def total_transactions_count(client_id: int = Field()):
        result = run_analysis(filters={"client_id": {"eq": client_id}}, metrics=["count(*)"], limit=1)
        if not result["data"]:
            return {"total_transactions": 0, "client_id": client_id}
        return {"total_transactions": int(result["data"][0]["count(*)"]), "client_id": client_id}

    @mcp.tool(name="average_transaction_amount", description="Average transaction amount.")
    def average_transaction_amount(client_id: int = Field()):
        result = run_analysis(filters={"client_id": {"eq": client_id}}, metrics=["avg(amount)"], limit=1)
        if not result["data"]:
            return error_response(f"No transactions for client {client_id}")
        return {"average_amount": float(result["data"][0]["avg(amount)"]), "client_id": client_id}

    @mcp.tool(name="spending_forecast", description="Forecast spending for next N months.")
    def spending_forecast(client_id: int = Field(), months: int = Field(default=3)):
        result = run_forecast_analysis(client_id=client_id, months=months)
        if "error" in result:
            return error_response(result["error"])
        return result

    @mcp.tool(name="transaction_heatmap", description="Heatmap of spending by day/hour.")
    def transaction_heatmap(client_id: int = Field()):
        df = get_transactions()
        df = df[df["client_id"] == client_id].copy()
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
        df = get_transactions()
        df = df[df["client_id"] == client_id].copy()
        if df.empty:
            return error_response(f"No transactions for client {client_id}")
        df["hour"] = df["timestamp"].dt.hour
        df["day_name"] = df["timestamp"].dt.day_name()
        peak_hours = df.groupby("hour")["amount"].sum().sort_values(ascending=False).head(5)
        peak_days = df.groupby("day_name")["amount"].sum().sort_values(ascending=False)
        return {"peak_hours": {int(k): float(v) for k, v in peak_hours.items()}, "peak_days": {k: float(v) for k, v in peak_days.items()}}

    @mcp.tool(name="spending_distribution", description="Distribution of transaction amounts.")
    def spending_distribution(client_id: int = Field()):
        df = get_transactions()
        df = df[df["client_id"] == client_id]
        if df.empty:
            return error_response(f"No transactions for client {client_id}")
        amounts = df["amount"].describe()
        return {
            "count": int(amounts["count"]),
            "min": float(amounts["min"]),
            "max": float(amounts["max"]),
            "mean": float(amounts["mean"]),
            "median": float(amounts["50%"]),
            "std": float(amounts["std"]),
        }

    @mcp.tool(name="merchant_summary", description="Merchant summary by MCC code.")
    def merchant_summary(mcc_code: str = Field()):
        filter_value = int(mcc_code) if str(mcc_code).isdigit() else mcc_code
        result = run_analysis(
            filters={"mcc_code": {"eq": filter_value}},
            metrics=["count(*)", "sum(amount)", "avg(amount)", "fraud_rate"],
            limit=1,
        )
        if not result["data"]:
            return error_response(f"No data for MCC {mcc_code}")
        row = result["data"][0]
        return {
            "mcc_code": mcc_code,
            "total_transactions": int(row["count(*)"]),
            "total_revenue": float(row["sum(amount)"]),
            "avg_transaction": float(row["avg(amount)"]),
            "fraud_rate": float(row["fraud_rate"]),
        }

    @mcp.tool(name="merchant_risk_analysis", description="Analyze merchant risk level.")
    def merchant_risk_analysis(mcc_code: str = Field()):
        df = get_transactions()
        df = df[df["mcc_code"].astype(str) == mcc_code]
        if df.empty:
            return error_response(f"No data for MCC {mcc_code}")
        fraud_rate = df["is_fraud"].mean() if "is_fraud" in df.columns else 0
        avg_amount = df["amount"].mean()
        return {"mcc_code": mcc_code, "fraud_rate": round(float(fraud_rate), 3), "avg_amount": float(avg_amount), "risk_level": "high" if fraud_rate > 0.1 else "medium" if fraud_rate > 0.05 else "low"}

    @mcp.tool(name="top_merchants", description="Top merchants by spending.")
    def top_merchants(client_id: int = Field(), limit: int = Field(default=5)):
        result = run_analysis(
            filters={"client_id": {"eq": client_id}},
            group_by=["merchant_category"],
            metrics=["sum(amount)"],
            sort_by="sum(amount)",
            limit=limit,
        )
        if not result["data"]:
            return error_response(f"No transactions for client {client_id}")
        return {"merchants": {item["merchant_category"]: float(item["sum(amount)"]) for item in result["data"]}}

    @mcp.tool(name="card_usage_analysis", description="Card usage patterns.")
    def card_usage_analysis(client_id: int = Field()):
        df = get_transactions()
        df = df[df["client_id"] == client_id]
        if df.empty:
            return error_response(f"No transactions for client {client_id}")
        chip_usage = df.groupby("use_chip")["amount"].sum()
        card_types = df.groupby("card_type")["amount"].sum()
        card_brands = df.groupby("card_brand")["amount"].sum()
        return {"chip_vs_swipe": {str(k): float(v) for k, v in chip_usage.items()}, "by_card_type": {k: float(v) for k, v in card_types.items()}, "by_card_brand": {k: float(v) for k, v in card_brands.items()}}

    @mcp.tool(name="card_type_spending", description="Spending by card brand.")
    def card_type_spending(client_id: int = Field()):
        df = get_transactions()
        df = df[df["client_id"] == client_id]
        if df.empty:
            return error_response(f"No transactions for client {client_id}")
        brands = df.groupby("card_brand")["amount"].sum().sort_values(ascending=False)
        return {"brands": {k: float(v) for k, v in brands.items()}}

    @mcp.tool(name="customer_profile", description="Complete customer profile.")
    def customer_profile(client_id: int = Field()):
        result = get_client_overview(client_id)
        if "error" in result:
            return error_response(result["error"])
        return result

    @mcp.tool(name="income_analysis", description="Income and spending ratio analysis.")
    def income_analysis(client_id: int = Field()):
        result = get_client_overview(client_id)
        if "error" in result:
            return error_response(result["error"])
        return {
            "client_id": client_id,
            "yearly_income": result["yearly_income"],
            "total_spent": result["total_spent"],
            "spending_to_income_ratio": result["spending_to_income_ratio"],
            "status": "high" if result["spending_to_income_ratio"] > 0.5 else "moderate" if result["spending_to_income_ratio"] > 0.25 else "low",
        }
