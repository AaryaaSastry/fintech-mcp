"""LLM-backed tool planner with validation and safe fallback handling."""
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

MODEL_ID = "gemma-3-27b-it"


def _get_client() -> genai.Client:
    """Build a fresh Gemini client from current environment configuration."""
    return genai.Client()


def _generate_content(model_id: str, prompt: str):
    """Run a Gemini request with an explicit per-call client lifecycle."""
    client = _get_client()
    try:
        return client.models.generate_content(model=model_id, contents=prompt)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


class ToolPlanner:
    """Plans MCP tool calls from natural language queries."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id

    def _build_prompt(
        self,
        query: str,
        tools_metadata: List[Dict[str, Any]],
        client_name_map: Dict[str, int],
    ) -> str:
        tools_json = json.dumps(tools_metadata, indent=2)
        client_json = json.dumps(client_name_map, indent=2, sort_keys=True)

        return f"""You are a tool planner for a fintech analytics assistant.

Your job is to choose the correct analytics tools for the user's request.
Do not answer the user. Only return JSON.

Available tools:
{tools_json}

Known client name map:
{client_json}

Rules:
1. Return valid JSON only.
2. Use only the available tool names.
3. Prefer the smallest tool plan that answers the user's query.
4. If the user asks for a forecast or future spending, choose a forecast tool.
5. If the user asks for a trend over time, choose a trend tool.
6. Respect time windows like "last month", "3 months", "quarter", "year".
7. Handle common misspellings such as "forcast" as "forecast".
8. If the query is about all users, use aggregate tools.
9. If a client name is mentioned, resolve it to the mapped client_id.
10. Only use a numeric client_id when the query explicitly says client/user/customer plus the number.
11. Prefer the generic tools over narrow legacy tools.
12. For grouped spending analysis use run_aggregate_analysis with metrics like sum(amount), avg(amount), count(*), fraud_rate, distinct_count(client_id), or distinct_count(merchant_id).
13. For trends use run_aggregate_analysis with time_grain="month" and sort_by="time_grain".
14. For forecasts use run_forecast_analysis.
15. For client credit/income/debt/profile questions use get_client_overview.
16. For fraud questions use run_fraud_analysis with the appropriate mode.
17. If you are unsure, return an empty steps list.

Examples:
- "show client 3 spending by city" -> run_aggregate_analysis with filters={{"client_id": {{"eq": 3}}}}, group_by=["merchant_city"], metrics=["sum(amount)", "count(*)"], sort_by="sum(amount)"
- "predict what all users will spend in the next 3 months" -> run_forecast_analysis with client_id omitted and months=3
- "what is client 4 credit score" -> get_client_overview with client_id=4
- "is transaction 52 fraudulent" -> run_fraud_analysis with mode="transaction_check" and transaction_id=52

Output schema:
{{
  "steps": [
    {{
      "tool": "tool_name",
      "params": {{}}
    }}
  ]
}}

User query:
{query}
"""

    def _build_clarification_prompt(
        self,
        query: str,
        reason: str,
        query_metadata: Dict[str, Any],
    ) -> str:
        metadata_json = json.dumps(query_metadata, indent=2, sort_keys=True)
        return f"""You are a clarification generator for a fintech analytics assistant.

The user's request cannot be answered directly from the available dataset.
Your job is to propose exactly 3 plausible interpretations of what the user may have meant.
Do not answer the original request. Only return JSON.

Context:
- User query: {query}
- Why clarification is needed: {reason}
- Query metadata: {metadata_json}

Rules:
1. Return valid JSON only.
2. Return exactly 3 options.
3. Each option must be realistic, specific, and close to the user's wording.
4. Each option must include:
   - id: "1", "2", or "3"
   - label: short user-facing clarification text
   - query: a rewritten direct query the system can run
5. Do not invent unavailable data such as credit-score-over-time if the reason says that data is missing.
6. Prefer alternatives that the current dataset can support, such as current profile fields, spending history, fraud summary, or credit/income overview.

Output schema:
{{
  "options": [
    {{
      "id": "1",
      "label": "short option label",
      "query": "direct rewritten query"
    }},
    {{
      "id": "2",
      "label": "short option label",
      "query": "direct rewritten query"
    }},
    {{
      "id": "3",
      "label": "short option label",
      "query": "direct rewritten query"
    }}
  ]
}}
"""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Planner response did not contain JSON")
        return json.loads(text[start:end + 1])

    def _coerce_scalar(self, value: Any, expected_type: str) -> Any:
        if value is None:
            return None
        if expected_type == "int":
            return int(value)
        if expected_type == "float":
            return float(value)
        if expected_type == "str":
            return str(value)
        return value

    def validate_plan(self, plan: Dict[str, Any], tools_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tool_map = {tool["name"]: tool.get("params", {}) for tool in tools_metadata}
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("Planner output must contain a list of steps")

        validated_steps: List[Dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Each planner step must be an object")

            tool_name = step.get("tool")
            params = step.get("params", {})
            if tool_name not in tool_map:
                raise ValueError(f"Planner selected unknown tool '{tool_name}'")
            if not isinstance(params, dict):
                raise ValueError(f"Planner params for '{tool_name}' must be an object")

            expected_params = tool_map[tool_name]
            cleaned_params = {}
            for key, value in params.items():
                if key not in expected_params:
                    raise ValueError(f"Planner provided invalid param '{key}' for tool '{tool_name}'")
                cleaned_params[key] = self._coerce_scalar(value, expected_params[key])

            validated_steps.append({"tool": tool_name, "params": cleaned_params})

        return validated_steps

    async def plan_query(
        self,
        query: str,
        tools_metadata: List[Dict[str, Any]],
        client_name_map: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(query, tools_metadata, client_name_map)
        response = _generate_content(self.model_id, prompt)
        response_text = response.text or ""
        plan = self._extract_json(response_text)
        return self.validate_plan(plan, tools_metadata)

    async def generate_clarification_options(
        self,
        query: str,
        reason: str,
        query_metadata: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        prompt = self._build_clarification_prompt(query, reason, query_metadata)
        response = _generate_content(self.model_id, prompt)
        payload = self._extract_json(response.text or "")
        options = payload.get("options", [])
        if not isinstance(options, list) or len(options) != 3:
            raise ValueError("Clarification response must contain exactly 3 options")

        cleaned_options: List[Dict[str, str]] = []
        for expected_id, option in zip(["1", "2", "3"], options):
            if not isinstance(option, dict):
                raise ValueError("Each clarification option must be an object")
            label = str(option.get("label", "")).strip()
            rewritten_query = str(option.get("query", "")).strip()
            if not label or not rewritten_query:
                raise ValueError("Clarification options must include label and query")
            cleaned_options.append(
                {
                    "id": expected_id,
                    "label": label,
                    "query": rewritten_query,
                }
            )
        return cleaned_options


tool_planner = ToolPlanner()
