"""MCP server entrypoint for fintech analytics."""
from mcp.server.fastmcp import FastMCP

from mcp_server.tools.aggregate_runtime import register_tools as register_aggregate_tools
from mcp_server.tools.client_risk import register_tools as register_client_risk_tools
from mcp_server.tools.client_spending import register_tools as register_client_spending_tools
from mcp_server.tools.generic_analytics import register_tools as register_generic_analytics_tools


mcp = FastMCP("FintechAnalytics", log_level="ERROR")

register_generic_analytics_tools(mcp)
register_client_spending_tools(mcp)
register_client_risk_tools(mcp)
register_aggregate_tools(mcp)


if __name__ == "__main__":
    mcp.run(transport="stdio")
