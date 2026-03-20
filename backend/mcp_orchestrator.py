"""Routes user queries to generic MCP analytics tools with LLM planning and fallback."""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from .mcp_client import call_mcp_tool
    from .tool_planner import tool_planner
except ImportError:
    from mcp_client import call_mcp_tool
    from tool_planner import tool_planner

load_dotenv()

CLIENT_NAME_MAP = {
    "soniya": 4,
    "sonia": 4,
    "john": 11,
    "emma": 14,
    "mike": 20,
    "sarah": 7,
    "ava": 21,
    "liam": 22,
    "noah": 23,
    "mia": 24,
    "zoe": 25,
    "ethan": 26,
    "nora": 27,
    "lucas": 28,
    "isla": 29,
    "owen": 30,
}

TOOLS_METADATA = [
    {
        "name": "get_analytics_schema",
        "description": "Returns supported columns, metrics, operators, and time grains for generic analysis.",
        "params": {},
    },
    {
        "name": "run_aggregate_analysis",
        "description": "General transaction analytics tool for totals, grouped breakdowns, averages, counts, merchant/category/city/card analysis, and time series summaries.",
        "params": {
            "filters": "dict",
            "group_by": "list",
            "metrics": "list",
            "sort_by": "str",
            "order": "str",
            "limit": "int",
            "time_grain": "str",
            "recent_months": "int",
        },
    },
    {
        "name": "run_forecast_analysis",
        "description": "Average-based spending projection for a single client or all users.",
        "params": {"client_id": "int", "months": "int"},
    },
    {
        "name": "get_client_overview",
        "description": "Returns a client's spending, income, debt, credit score, utilization, and profile overview.",
        "params": {"client_id": "int"},
    },
    {
        "name": "run_fraud_analysis",
        "description": "Fraud analysis tool. Modes: global_summary, client_summary, transaction_check, merchant_summary, high_risk_transactions.",
        "params": {
            "mode": "str",
            "client_id": "int",
            "transaction_id": "int",
            "mcc_code": "str",
            "threshold": "float",
        },
    },
]


@lru_cache(maxsize=1)
def _load_transactions() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
    return pd.read_csv(data_path, engine="python", on_bad_lines="skip")


def client_exists(client_id: int) -> bool:
    try:
        return client_id in _load_transactions()["client_id"].values
    except Exception:
        return False


def get_transaction_for_client(client_id: int | None) -> int | None:
    if client_id is None:
        return None

    try:
        client_txs = _load_transactions().loc[lambda df: df["client_id"] == client_id]
        if not client_txs.empty:
            return int(client_txs["transaction_id"].iloc[-1])
    except Exception:
        return None
    return None


def fix_param_types(params: dict) -> dict:
    """Convert string params to proper scalar types without mutating nested values."""
    fixed = {}
    int_params = {"client_id", "transaction_id", "months", "limit", "recent_months"}
    float_params = {"threshold"}

    for key, value in params.items():
        if key in int_params and isinstance(value, str):
            try:
                fixed[key] = int(value)
            except ValueError:
                fixed[key] = value
        elif key in float_params and isinstance(value, str):
            try:
                fixed[key] = float(value)
            except ValueError:
                fixed[key] = value
        else:
            fixed[key] = value
    return fixed


def extract_month_window(query: str, default: int = 3) -> int:
    q = query.lower()

    if any(phrase in q for phrase in ["last month", "this month", "past month", "previous month"]):
        return 1

    explicit = re.search(r"(\d+)\s*(month|months)", q)
    if explicit:
        return max(1, int(explicit.group(1)))

    if any(phrase in q for phrase in ["quarter", "last quarter", "past quarter"]):
        return 3

    if "half year" in q or "six months" in q:
        return 6

    if "year" in q or "12 months" in q:
        return 12

    return default


def has_explicit_time_window(query: str) -> bool:
    q = query.lower()
    return bool(
        re.search(r"\d+\s*(month|months)", q)
        or any(phrase in q for phrase in ["last month", "this month", "past month", "previous month", "quarter", "year", "six months", "half year"])
    )


class MCPOrchestrator:
    def __init__(self):
        self.tools_metadata = TOOLS_METADATA
        self._all_users_phrases = [
            "all users",
            "all clients",
            "everyone",
            "overall",
            "global",
            "aggregate",
            "across all",
            "all the users",
            "all customers",
        ]
        self._non_timeseries_profile_fields = {
            "credit": "credit score",
            "credit score": "credit score",
            "income": "income",
            "debt": "debt",
            "credit limit": "credit limit",
            "age": "age",
        }

    def _extract_explicit_client_id(self, query: str) -> int | None:
        match = re.search(r"\b(?:client|user|customer)\s*(?:id\s*)?(\d+)\b", query.lower())
        if not match:
            return None
        client_id = int(match.group(1))
        return client_id if client_exists(client_id) else None

    def _extract_explicit_transaction_id(self, query: str) -> int | None:
        match = re.search(r"\btransaction\s*(?:id\s*)?(\d+)\b", query.lower())
        if not match:
            return None
        return int(match.group(1))

    def _resolve_query_context(self, query: str) -> dict:
        q = query.lower()
        is_all_users = any(phrase in q for phrase in self._all_users_phrases)
        client_id = None
        user_name = None

        for name, mapped_id in CLIENT_NAME_MAP.items():
            if name in q:
                client_id = mapped_id
                user_name = name
                break

        if client_id is None:
            client_id = self._extract_explicit_client_id(query)

        explicit_transaction_id = self._extract_explicit_transaction_id(query)

        if client_id is None and explicit_transaction_id is None and not is_all_users:
            is_all_users = True

        if client_id is not None and not client_exists(client_id):
            client_id = None
            user_name = None
            is_all_users = True

        return {
            "is_all_users": is_all_users,
            "client_id": client_id,
            "user_name": user_name,
            "transaction_id": explicit_transaction_id if explicit_transaction_id is not None else (None if is_all_users else get_transaction_for_client(client_id)),
        }

    def _base_filters(self, query_metadata: dict) -> dict | None:
        if query_metadata["is_all_users"]:
            return None
        return {"client_id": {"eq": query_metadata["client_id"]}}

    def _clarification_reason(self, query: str, query_metadata: dict) -> str | None:
        """Explain why clarification is needed when the dataset cannot support the requested shape directly."""
        q = query.lower()
        wants_time_series = any(
            phrase in q
            for phrase in ["history", "trend", "over time", "line graph", "line chart", "graph", "chart"]
        )
        matched_field = None
        for phrase, normalized in self._non_timeseries_profile_fields.items():
            if phrase in q:
                matched_field = normalized
                break

        if not wants_time_series or matched_field is None:
            return None

        if query_metadata["client_id"] is None:
            return None

        client_id = query_metadata["client_id"]
        return f"The dataset does not contain {matched_field} snapshots over time for client {client_id}."

    def _aggregate_step(
        self,
        query_metadata: dict,
        metrics: list[str],
        group_by: list[str] | None = None,
        sort_by: str | None = None,
        order: str = "desc",
        limit: int = 10,
        time_grain: str | None = None,
        recent_months: int | None = None,
    ) -> dict:
        effective_group_by = list(group_by or [])
        if query_metadata["is_all_users"] and "client_id" not in effective_group_by:
            effective_group_by.insert(0, "client_id")

        effective_sort_by = sort_by
        if query_metadata["is_all_users"] and effective_sort_by == "sum(amount)":
            effective_sort_by = "client_id"
        if query_metadata["is_all_users"] and effective_sort_by is None and "client_id" in effective_group_by:
            effective_sort_by = "client_id"

        effective_limit = max(limit, 100) if query_metadata["is_all_users"] else limit

        return {
            "tool": "run_aggregate_analysis",
            "params": {
                "filters": self._base_filters(query_metadata),
                "group_by": effective_group_by,
                "metrics": metrics,
                "sort_by": effective_sort_by,
                "order": order,
                "limit": effective_limit,
                "time_grain": time_grain,
                "recent_months": recent_months,
            },
        }

    def _keyword_fallback(self, query: str, query_metadata: dict) -> list:
        q = query.lower()
        months_window = extract_month_window(query, default=3)
        recent_months = months_window if has_explicit_time_window(query) or any(word in q for word in ["spend", "spent", "summary", "forecast", "forcast", "future", "predict"]) else None
        client_id = query_metadata["client_id"]
        transaction_id = query_metadata["transaction_id"]

        forecast_requested = any(word in q for word in ["forecast", "forcast", "future", "predict", "prediction"])
        fraud_requested = any(word in q for word in ["fraud", "risky", "risk", "suspicious"])
        trend_requested = any(word in q for word in ["trend", "history"]) or ("month" in q and not forecast_requested)
        profile_requested = any(word in q for word in ["profile", "credit", "income", "debt", "limit", "overview"])
        category_requested = any(word in q for word in ["category", "categories"])
        city_requested = any(word in q for word in ["city", "location"])
        card_requested = "card" in q
        merchant_requested = "merchant" in q
        count_requested = "count" in q or "how many" in q
        average_requested = "average" in q or "avg" in q
        spending_requested = any(word in q for word in ["spend", "spent", "summary", "total"])

        plan = []

        if forecast_requested:
            plan.append(
                {
                    "tool": "run_forecast_analysis",
                    "params": {"client_id": client_id, "months": months_window},
                }
            )

        if fraud_requested:
            if query_metadata["is_all_users"]:
                plan.append({"tool": "run_fraud_analysis", "params": {"mode": "global_summary"}})
            else:
                if transaction_id is not None and any(word in q for word in ["transaction", "suspicious"]):
                    plan.append(
                        {
                            "tool": "run_fraud_analysis",
                            "params": {"mode": "transaction_check", "transaction_id": transaction_id},
                        }
                    )
                elif client_id is not None:
                    plan.append(
                        {
                            "tool": "run_fraud_analysis",
                            "params": {"mode": "client_summary", "client_id": client_id},
                        }
                    )

        if profile_requested and not query_metadata["is_all_users"]:
            plan.append({"tool": "get_client_overview", "params": {"client_id": client_id}})

        if category_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["sum(amount)", "count(*)"],
                    group_by=["merchant_category"],
                    sort_by="sum(amount)",
                    recent_months=recent_months,
                )
            )

        if city_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["sum(amount)", "count(*)"],
                    group_by=["merchant_city"],
                    sort_by="sum(amount)",
                    recent_months=recent_months,
                )
            )

        if card_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["sum(amount)", "count(*)"],
                    group_by=["card_type"],
                    sort_by="sum(amount)",
                    recent_months=recent_months,
                )
            )

        if merchant_requested and not category_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["sum(amount)", "count(*)"],
                    group_by=["merchant_category"],
                    sort_by="sum(amount)",
                    recent_months=recent_months,
                )
            )

        if trend_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["sum(amount)", "count(*)"],
                    sort_by="time_grain",
                    order="asc",
                    time_grain="month",
                    limit=60,
                    recent_months=recent_months,
                )
            )

        if spending_requested and not any(requested for requested in [category_requested, city_requested, card_requested, merchant_requested, trend_requested, forecast_requested]):
            metrics = ["sum(amount)", "count(*)"]
            if average_requested:
                metrics.append("avg(amount)")
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=metrics,
                    limit=1,
                    recent_months=recent_months or months_window,
                )
            )

        if count_requested and not spending_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["count(*)"],
                    limit=1,
                    recent_months=recent_months,
                )
            )

        if average_requested and not spending_requested:
            plan.append(
                self._aggregate_step(
                    query_metadata,
                    metrics=["avg(amount)"],
                    limit=1,
                    recent_months=recent_months,
                )
            )

        if not plan:
            plan.append({"tool": "get_analytics_schema", "params": {}})

        return plan

    def _should_skip_planner(self, query: str, fallback_plan: list) -> bool:
        """Use the fast path for obvious intents that the fallback handles well."""
        if not fallback_plan:
            return False
        if fallback_plan == [{"tool": "get_analytics_schema", "params": {}}]:
            return False

        q = query.lower()
        slow_planner_signals = [
            "compare",
            "versus",
            "vs",
            "correlation",
            "segment",
            "cohort",
            "why",
            "explain",
            "reason",
            "anomaly",
        ]
        return not any(signal in q for signal in slow_planner_signals)

    async def _plan_with_llm(self, query: str) -> list:
        return await tool_planner.plan_query(query, self.tools_metadata, CLIENT_NAME_MAP)

    async def handle_query(self, user_query: str):
        query_metadata = self._resolve_query_context(user_query)
        clarification_reason = self._clarification_reason(user_query, query_metadata)
        if clarification_reason is not None:
            try:
                options = await tool_planner.generate_clarification_options(
                    user_query,
                    clarification_reason,
                    {
                        "is_all_users": query_metadata["is_all_users"],
                        "client_id": query_metadata["client_id"],
                        "user_name": query_metadata["user_name"],
                    },
                )
                return {
                    "type": "clarification",
                    "clarification": {
                        "reason": clarification_reason,
                        "options": options,
                    },
                    "query_metadata": {
                        "is_all_users": query_metadata["is_all_users"],
                        "client_id": query_metadata["client_id"],
                        "user_name": query_metadata["user_name"],
                    },
                }
            except Exception as exc:
                return {
                    "type": "clarification",
                    "clarification": {
                        "reason": clarification_reason,
                        "options": [],
                        "fallback_message": f"I need a bit more direction before I run anything. Clarification generation failed: {exc}",
                    },
                    "query_metadata": {
                        "is_all_users": query_metadata["is_all_users"],
                        "client_id": query_metadata["client_id"],
                        "user_name": query_metadata["user_name"],
                    },
                }

        fallback_plan = self._keyword_fallback(user_query, query_metadata)
        planner_used = False

        if self._should_skip_planner(user_query, fallback_plan):
            plan = fallback_plan
        else:
            try:
                planned_steps = await self._plan_with_llm(user_query)
                plan = planned_steps if planned_steps else fallback_plan
                planner_used = bool(planned_steps)
            except Exception as exc:
                print(f"Planner fallback triggered: {exc}")
                plan = fallback_plan

        print(f"Plan: {plan}")
        print(f"Query Metadata: {query_metadata}")

        results = {}
        for step in plan:
            tool_name = step.get("tool")
            params = fix_param_types(step.get("params", {}))
            print(f"Executing: {tool_name} with {params}")
            try:
                results[tool_name] = await call_mcp_tool(tool_name, params)
            except Exception as exc:
                results[tool_name] = {"error": str(exc)}

        if not results:
            return {"summary": "No data found.", "query_metadata": query_metadata}

        try:
            summary_parts = []
            for tool_name, result in results.items():
                if isinstance(result, dict) and "error" not in result:
                    if "data" in result and "meta" in result:
                        summary_parts.append(f"{tool_name}: {result['meta'].get('row_count', 0)} rows")
                    elif "forecast" in result:
                        summary_parts.append(f"{tool_name}: forecast ready")
                    elif "risk_score" in result:
                        summary_parts.append(f"{tool_name}: risk score {result.get('risk_score')}")
                    else:
                        summary_parts.append(f"{tool_name}: {len(result)} fields")
                elif isinstance(result, dict):
                    summary_parts.append(f"{tool_name}: {result.get('error')}")

            summary = "; ".join(summary_parts) if summary_parts else "Results available"
        except Exception:
            summary = " | ".join(results.keys())

        return {
            "summary": summary,
            "raw_data": results,
            "plan": plan,
            "planner_used": planner_used,
            "query_metadata": {
                "is_all_users": query_metadata["is_all_users"],
                "client_id": query_metadata["client_id"],
                "user_name": query_metadata["user_name"],
            },
        }


orchestrator = MCPOrchestrator()
