"""Client fraud and credit MCP tools."""

from pydantic import Field

from engine.fraud import (
    check_transaction_fraud,
    get_client_fraud_risk,
    get_high_risk_transactions,
    get_merchant_fraud_summary,
)
from engine.profiles import get_client_overview

from .common import error_response


def register_tools(mcp):
    @mcp.tool(name="check_fraud", description="Estimate fraud probability for a transaction using ML model.")
    def check_fraud(transaction_id: int = Field()):
        return check_transaction_fraud(transaction_id)

    @mcp.tool(name="client_fraud_risk", description="Calculate overall fraud risk score for a client using ML model.")
    def client_fraud_risk(client_id: int = Field()):
        return get_client_fraud_risk(client_id)

    @mcp.tool(name="fraud_rate_by_merchant", description="Fraud rate by merchant category.")
    def fraud_rate_by_merchant(mcc_code: str = Field()):
        return get_merchant_fraud_summary(mcc_code)

    @mcp.tool(name="high_risk_transactions", description="List high-value transactions above threshold.")
    def high_risk_transactions(client_id: int = Field(), threshold: float = Field(default=500)):
        return get_high_risk_transactions(client_id, threshold)

    @mcp.tool(name="credit_score_history", description="Credit score with grade.")
    def credit_score_history(client_id: int = Field()):
        overview = get_client_overview(client_id)
        if "error" in overview:
            return error_response(overview["error"])
        return {"client_id": client_id, "credit_score": overview["credit_score"], "grade": overview["credit_grade"]}

    @mcp.tool(name="debt_income_analysis", description="Debt to income ratio analysis.")
    def debt_income_analysis(client_id: int = Field()):
        overview = get_client_overview(client_id)
        if "error" in overview:
            return error_response(overview["error"])
        return {
            "client_id": client_id,
            "total_debt": overview["total_debt"],
            "yearly_income": overview["yearly_income"],
            "debt_to_income_ratio": overview["debt_to_income_ratio"],
            "risk_level": overview["debt_risk_level"],
        }

    @mcp.tool(name="credit_limit_analysis", description="Credit limit and utilization analysis.")
    def credit_limit_analysis(client_id: int = Field()):
        overview = get_client_overview(client_id)
        if "error" in overview:
            return error_response(overview["error"])
        return {
            "client_id": client_id,
            "credit_limit": overview["credit_limit"],
            "total_spent": overview["total_spent"],
            "utilization_percent": overview["utilization_percent"],
            "status": overview["utilization_status"],
        }
