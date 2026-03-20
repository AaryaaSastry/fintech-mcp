"""Generic MCP tools backed by the shared analytics engine."""

from pydantic import Field

from engine.analysis_service import run_analysis
from engine.forecasting import run_forecast_analysis
from engine.fraud import run_fraud_analysis
from engine.profiles import get_client_overview
from engine.validators import ALLOWED_COLUMNS, ALLOWED_OPERATORS, ALLOWED_TIME_GRAINS
from engine.metrics import list_supported_metrics


def register_tools(mcp):
    @mcp.tool(
        name="get_analytics_schema",
        description="Return supported columns, metrics, operators, and time grains for generic analytics.",
    )
    def get_analytics_schema():
        return {
            "columns": sorted(ALLOWED_COLUMNS),
            "metrics": list_supported_metrics(),
            "operators": sorted(ALLOWED_OPERATORS),
            "time_grains": sorted(ALLOWED_TIME_GRAINS),
        }

    @mcp.tool(
        name="run_aggregate_analysis",
        description="Generic transaction analytics tool. Use for totals, averages, grouped breakdowns, and time series summaries.",
    )
    def run_aggregate_analysis(
        filters: dict | None = Field(default=None),
        group_by: list[str] | None = Field(default=None),
        metrics: list[str] = Field(default_factory=lambda: ["count(*)"]),
        sort_by: str | None = Field(default=None),
        order: str = Field(default="desc"),
        limit: int = Field(default=100),
        time_grain: str | None = Field(default=None),
        recent_months: int | None = Field(default=None),
    ):
        return run_analysis(
            filters=filters,
            group_by=group_by,
            metrics=metrics,
            sort_by=sort_by,
            order=order,
            limit=limit,
            time_grain=time_grain,
            recent_months=recent_months,
        )

    @mcp.tool(
        name="run_forecast_analysis",
        description="Average-based spending projection for a client or all users.",
    )
    def run_forecast_tool(
        client_id: int | None = Field(default=None),
        months: int = Field(default=3),
    ):
        return run_forecast_analysis(client_id=client_id, months=months)

    @mcp.tool(
        name="get_client_overview",
        description="Return a client's spending, credit, debt, utilization, and income overview.",
    )
    def get_client_overview_tool(client_id: int = Field()):
        return get_client_overview(client_id)

    @mcp.tool(
        name="run_fraud_analysis",
        description="Generic fraud analysis tool. Modes: global_summary, client_summary, transaction_check, merchant_summary, high_risk_transactions.",
    )
    def run_fraud_tool(
        mode: str = Field(),
        client_id: int | None = Field(default=None),
        transaction_id: int | None = Field(default=None),
        mcc_code: str | None = Field(default=None),
        threshold: float = Field(default=500.0),
    ):
        return run_fraud_analysis(
            mode=mode,
            client_id=client_id,
            transaction_id=transaction_id,
            mcc_code=mcc_code,
            threshold=threshold,
        )
