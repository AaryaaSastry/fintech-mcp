import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.mcp_orchestrator import CLIENT_NAME_MAP, MCPOrchestrator, extract_month_window
from backend.tool_planner import ToolPlanner


TOOLS_METADATA = [
    {"name": "run_forecast_analysis", "params": {"client_id": "int", "months": "int"}},
    {"name": "run_aggregate_analysis", "params": {"filters": "dict", "group_by": "list", "metrics": "list"}},
]


class PlannerAndOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.planner = ToolPlanner()
        self.orchestrator = MCPOrchestrator()

    def test_extract_month_window_handles_common_phrases(self):
        self.assertEqual(extract_month_window("show spending last month"), 1)
        self.assertEqual(extract_month_window("forecast for the next 6 months"), 6)
        self.assertEqual(extract_month_window("show spending this quarter"), 3)

    def test_keyword_fallback_handles_forecast_misspelling(self):
        metadata = {
            "is_all_users": False,
            "client_id": 3,
            "user_name": None,
            "transaction_id": 3,
        }
        plan = self.orchestrator._keyword_fallback(
            "forcast client 3 spending for the next 3 months",
            metadata,
        )

        self.assertIn(
            {"tool": "run_forecast_analysis", "params": {"client_id": 3, "months": 3}},
            plan,
        )

    def test_validate_plan_coerces_parameter_types(self):
        plan = {
            "steps": [
                {
                    "tool": "run_forecast_analysis",
                    "params": {"client_id": "3", "months": "6"},
                }
            ]
        }

        validated = self.planner.validate_plan(plan, TOOLS_METADATA)
        self.assertEqual(
            validated,
            [{"tool": "run_forecast_analysis", "params": {"client_id": 3, "months": 6}}],
        )

    def test_validate_plan_rejects_unknown_tool(self):
        plan = {"steps": [{"tool": "unknown_tool", "params": {}}]}

        with self.assertRaisesRegex(ValueError, "unknown tool"):
            self.planner.validate_plan(plan, TOOLS_METADATA)

    def test_query_context_resolves_named_clients(self):
        context = self.orchestrator._resolve_query_context("show Ava spending")

        self.assertFalse(context["is_all_users"])
        self.assertEqual(context["client_id"], CLIENT_NAME_MAP["ava"])

    def test_query_context_does_not_treat_time_window_as_client_id(self):
        context = self.orchestrator._resolve_query_context(
            "predict what all the users will spend in the next 3 months"
        )

        self.assertTrue(context["is_all_users"])
        self.assertIsNone(context["client_id"])

    def test_keyword_fallback_prefers_forecast_tool_for_all_users(self):
        metadata = {
            "is_all_users": True,
            "client_id": None,
            "user_name": None,
            "transaction_id": None,
        }
        plan = self.orchestrator._keyword_fallback(
            "predict what all the users will spend in the next 3 months",
            metadata,
        )

        self.assertEqual(
            plan,
            [{"tool": "run_forecast_analysis", "params": {"client_id": None, "months": 3}}],
        )

    def test_all_users_spending_groups_by_client(self):
        metadata = {
            "is_all_users": True,
            "client_id": None,
            "user_name": None,
            "transaction_id": None,
        }

        plan = self.orchestrator._keyword_fallback("show spending for all users", metadata)

        self.assertEqual(
            plan,
            [
                {
                    "tool": "run_aggregate_analysis",
                    "params": {
                        "filters": None,
                        "group_by": ["client_id"],
                        "metrics": ["sum(amount)", "count(*)"],
                        "sort_by": "client_id",
                        "order": "desc",
                        "limit": 100,
                        "time_grain": None,
                        "recent_months": 3,
                    },
                }
            ],
        )

    def test_query_context_extracts_explicit_transaction_id(self):
        context = self.orchestrator._resolve_query_context("is transaction 52 fraudulent")

        self.assertEqual(context["transaction_id"], 52)
        self.assertFalse(context["is_all_users"])

    def test_fast_path_skips_planner_for_simple_query(self):
        context = self.orchestrator._resolve_query_context("show client 3 spending by city")
        fallback_plan = self.orchestrator._keyword_fallback("show client 3 spending by city", context)

        self.assertTrue(self.orchestrator._should_skip_planner("show client 3 spending by city", fallback_plan))

    def test_clarification_reason_requested_for_unsupported_credit_history_series(self):
        context = self.orchestrator._resolve_query_context(
            "what is the credit history of client 1, i want a line graph for the data"
        )
        reason = self.orchestrator._clarification_reason(
            "what is the credit history of client 1, i want a line graph for the data",
            context,
        )

        self.assertIsNotNone(reason)
        self.assertIn("credit score snapshots over time", reason)

    def test_handle_query_uses_llm_generated_clarification_options(self):
        query = "what is the credit history of client 1, i want a line graph for the data"
        mock_options = [
            {"id": "1", "label": "Show the current credit score", "query": "show client 1 current credit score"},
            {"id": "2", "label": "Show spending history as a line graph", "query": "show client 1 spending history as a line graph"},
            {"id": "3", "label": "Show the credit and income overview", "query": "show client 1 credit and income overview"},
        ]

        with patch(
            "backend.mcp_orchestrator.tool_planner.generate_clarification_options",
            new=AsyncMock(return_value=mock_options),
        ):
            result = asyncio.run(self.orchestrator.handle_query(query))

        self.assertEqual(result["type"], "clarification")
        self.assertEqual(result["clarification"]["options"], mock_options)


if __name__ == "__main__":
    unittest.main()
