"""Creates readable conversational responses from tool outputs."""
from pathlib import Path

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


class AIResponseGenerator:
    def __init__(self):
        self.model_id = MODEL_ID

    def _build_prompt(self, query: str, results: dict, query_type: str, query_metadata: dict = None) -> str:
        results_text = []
        for tool_name, data in results.items():
            if isinstance(data, dict):
                items = []
                for k, v in data.items():
                    if not k.startswith("_") and k != "error":
                        items.append(f"  - {k}: {v}")
                if items:
                    results_text.append(f"\n{tool_name}:\n" + "\n".join(items))
            elif isinstance(data, list) and data:
                results_text.append(f"\n{tool_name}: {len(data)} items")

        results_str = "\n".join(results_text) if results_text else "No data available"
        user_info = ""
        if query_metadata:
            if query_metadata.get("is_all_users", False):
                user_info = "\nUSER CONTEXT: ALL USERS (Aggregate Data)"
            elif query_metadata.get("user_name"):
                user_info = f"\nUSER CONTEXT: User Name: {query_metadata.get('user_name')}, Client ID: {query_metadata.get('client_id')}"
            elif query_metadata.get("client_id"):
                user_info = f"\nUSER CONTEXT: Client ID: {query_metadata.get('client_id')}"

        return f"""You are a professional financial data analyst.
USER QUESTION: {query}
QUERY TYPE: {query_type}{user_info}

TOOL RESULTS:
{results_str}

INSTRUCTIONS:
1. Explain the answer in plain English.
2. Do not print raw dictionaries, Python objects, or key-value dumps.
3. Convert fraud_rate and fraud_probability decimals into percentages.
4. Keep it concise and natural.
5. Do not add 'User:' or 'Client ID:' lines.
6. Do not add a follow-up question.
"""

    def _format_percent(self, value: float) -> str:
        return f"{value * 100:.2f}%"

    def _format_currency(self, value: float) -> str:
        return f"${float(value):,.2f}"

    def _format_user_context(self, query_metadata: dict = None) -> str:
        if not query_metadata:
            return ""
        if query_metadata.get("is_all_users", False):
            return "This is based on all users."
        if query_metadata.get("user_name") and query_metadata.get("client_id"):
            return f"This is based on user {query_metadata['user_name']} (client {query_metadata['client_id']})."
        if query_metadata.get("client_id"):
            return f"This is based on client {query_metadata['client_id']}."
        return ""

    def _looks_like_raw_dump(self, text: str) -> bool:
        markers = [
            "{'",
            "top_fraudulent_clients:",
            "fraud_count:",
            "total_transactions:",
            "User:",
            "Client ID:",
        ]
        return any(marker in text for marker in markers)

    def _find_payload(self, results: dict, predicate) -> dict | None:
        for value in results.values():
            if isinstance(value, dict) and predicate(value):
                return value
        return None

    def _is_fraud_summary_payload(self, payload: dict) -> bool:
        return (
            isinstance(payload.get("top_fraudulent_clients"), list)
            and "fraud_count" in payload
            and "fraud_rate" in payload
        )

    def _is_client_category_table(self, payload: dict) -> bool:
        rows = payload.get("data")
        return (
            isinstance(rows, list)
            and bool(rows)
            and all(isinstance(row, dict) and "client_id" in row and "merchant_category" in row for row in rows)
        )

    def _format_fraud_category_breakdown(self, results: dict) -> str | None:
        fraud_summary = self._find_payload(results, self._is_fraud_summary_payload)
        aggregate = self._find_payload(results, self._is_client_category_table)
        if fraud_summary is None or aggregate is None:
            return None

        top_clients = fraud_summary.get("top_fraudulent_clients")
        aggregate_rows = aggregate.get("data")
        if not isinstance(top_clients, list) or not isinstance(aggregate_rows, list):
            return None
        if not top_clients or not aggregate_rows:
            return None
        if not all(isinstance(row, dict) and "client_id" in row and "merchant_category" in row for row in aggregate_rows):
            return None

        top_client_ids = [int(item["client_id"]) for item in top_clients if isinstance(item, dict) and "client_id" in item]
        if not top_client_ids:
            return None

        rows_by_client: dict[int, list[dict]] = {client_id: [] for client_id in top_client_ids}
        for row in aggregate_rows:
            client_id = int(row["client_id"])
            if client_id in rows_by_client:
                rows_by_client[client_id].append(row)

        lines = []
        total_transactions = int(fraud_summary.get("total_transactions", 0))
        fraud_count = int(fraud_summary.get("fraud_count", 0))
        fraud_rate = float(fraud_summary.get("fraud_rate", 0))
        lines.append(
            f"There are {fraud_count} flagged transactions out of {total_transactions}, "
            f"for an overall fraud rate of {self._format_percent(fraud_rate)}."
        )
        lines.append("Top clients with flagged activity by category:")

        for item in top_clients:
            client_id = int(item["client_id"])
            client_rate = self._format_percent(float(item.get("fraud_rate", 0)))
            client_count = int(item.get("fraud_count", 0))
            categories = sorted(
                rows_by_client.get(client_id, []),
                key=lambda row: (-float(row.get("sum(amount)", 0)), str(row.get("merchant_category", ""))),
            )
            if not categories:
                continue

            category_parts = [
                f"{row['merchant_category']} ({self._format_currency(float(row.get('sum(amount)', 0)))}, {int(row.get('count(*)', 0))} transactions)"
                for row in categories[:4]
            ]
            lines.append(
                f"client {client_id}: fraud rate {client_rate}, {client_count} flagged transactions; "
                + "; ".join(category_parts)
            )

        return "\n".join(lines) if len(lines) > 2 else None

    def _format_client_rows(self, rows: list[dict], time_series: bool = False) -> str | None:
        if not rows or not all(isinstance(row, dict) and "client_id" in row for row in rows):
            return None

        grouped_rows: dict[int, list[dict]] = {}
        for row in rows:
            grouped_rows.setdefault(int(row["client_id"]), []).append(row)

        sections = []
        for client_id, client_rows in grouped_rows.items():
            if len(client_rows) == 1:
                row = client_rows[0]
                details = []
                if time_series and "time_grain" in row:
                    details.append(f"period {row['time_grain']}")
                if "sum(amount)" in row:
                    details.append(f"total {self._format_currency(float(row['sum(amount)']))}")
                if "avg(amount)" in row:
                    details.append(f"average {self._format_currency(float(row['avg(amount)']))}")
                if "count(*)" in row:
                    details.append(f"{int(row['count(*)'])} transactions")
                if "fraud_rate" in row:
                    details.append(f"fraud rate {self._format_percent(float(row['fraud_rate']))}")

                extra_dimensions = [
                    key for key in row.keys()
                    if key not in {"client_id", "time_grain", "sum(amount)", "avg(amount)", "count(*)", "fraud_rate", "min(amount)", "max(amount)", "distinct_count(client_id)", "distinct_count(merchant_id)"}
                ]
                for key in extra_dimensions[:2]:
                    details.append(f"{key} {row[key]}")

                sections.append(f"client {client_id}: " + (", ".join(details) if details else "data available"))
                continue

            dimension_keys = [
                key for key in client_rows[0].keys()
                if key not in {"client_id", "time_grain", "sum(amount)", "avg(amount)", "count(*)", "fraud_rate", "min(amount)", "max(amount)", "distinct_count(client_id)", "distinct_count(merchant_id)"}
            ]
            if len(dimension_keys) == 1:
                dimension = dimension_keys[0]
                ordered_rows = sorted(client_rows, key=lambda row: (-float(row.get("sum(amount)", 0)), str(row.get(dimension, ""))))
                dimension_parts = []
                for row in ordered_rows[:5]:
                    metrics = []
                    if "sum(amount)" in row:
                        metrics.append(self._format_currency(float(row["sum(amount)"])))
                    if "count(*)" in row:
                        metrics.append(f"{int(row['count(*)'])} transactions")
                    metric_text = ", ".join(metrics) if metrics else "data available"
                    dimension_parts.append(f"{row[dimension]} ({metric_text})")
                sections.append(f"client {client_id}: " + "; ".join(dimension_parts))
                continue

            sections.append(f"client {client_id}: {len(client_rows)} grouped rows")

        return "\n".join(sections)

    def _summarize_table_result(self, payload: dict) -> str | None:
        data = payload.get("data")
        meta = payload.get("meta", {})
        if not isinstance(data, list):
            return None
        if not data:
            return "No matching records were found."

        client_sections = self._format_client_rows(data, time_series=bool(data and isinstance(data[0], dict) and "time_grain" in data[0]))
        if client_sections:
            return client_sections

        if data and isinstance(data[0], dict) and "time_grain" in data[0] and len(data) > 1:
            latest = data[-1]
            parts = []
            if "sum(amount)" in latest:
                parts.append(f"${float(latest['sum(amount)']):,.2f} total")
            if "count(*)" in latest:
                parts.append(f"{int(latest['count(*)'])} transactions")
            detail = ", ".join(parts) if parts else "activity"
            return f"I found history across {len(data)} periods. The latest period is {latest['time_grain']} with {detail}."

        if data and isinstance(data[0], dict) and "time_grain" in data[0] and len(data) == 1:
            only = data[0]
            parts = []
            if "sum(amount)" in only:
                parts.append(f"${float(only['sum(amount)']):,.2f} total")
            if "count(*)" in only:
                parts.append(f"{int(only['count(*)'])} transactions")
            detail = ", ".join(parts) if parts else "activity"
            anchor = meta.get("recent_window_anchor")
            if meta.get("recent_months") and anchor:
                return (
                    f"I found 1 period in the recent {int(meta['recent_months'])}-month window, "
                    f"anchored to the latest available transaction date ({anchor}). "
                    f"That period is {only['time_grain']} with {detail}."
                )
            return f"I found 1 period. That period is {only['time_grain']} with {detail}."

        first_row = data[0]
        if not isinstance(first_row, dict):
            return f"I found {meta.get('row_count', len(data))} matching rows."

        parts = []
        if "sum(amount)" in first_row:
            parts.append(f"total amount {float(first_row['sum(amount)']):,.2f}")
        if "avg(amount)" in first_row:
            parts.append(f"average amount {float(first_row['avg(amount)']):,.2f}")
        if "count(*)" in first_row:
            parts.append(f"{int(first_row['count(*)'])} transactions")
        if "fraud_rate" in first_row:
            parts.append(f"fraud rate {self._format_percent(float(first_row['fraud_rate']))}")

        dimensions = [key for key in first_row.keys() if key not in {"sum(amount)", "avg(amount)", "count(*)", "fraud_rate", "min(amount)", "max(amount)", "distinct_count(client_id)", "distinct_count(merchant_id)"}]
        if dimensions:
            lead = ", ".join(f"{dimension} {first_row[dimension]}" for dimension in dimensions[:2])
            if parts:
                return f"The top result is {lead} with " + ", ".join(parts) + "."
            return f"The top result is {lead}."

        if parts:
            return "I found " + ", ".join(parts) + "."
        return f"I found {meta.get('row_count', len(data))} matching rows."

    async def generate_response(self, query: str, results: dict, query_type: str = "general", query_metadata: dict = None, prefer_llm: bool = True) -> str:
        if not results:
            return "I couldn't find any data for that."

        has_errors = all(isinstance(v, dict) and "error" in v for v in results.values()) if results else True
        if has_errors:
            return "I ran into some issues getting that information."

        if prefer_llm:
            try:
                prompt = self._build_prompt(query, results, query_type, query_metadata)
                response = _generate_content(self.model_id, prompt)
                if response.text:
                    text = response.text.strip()
                    if not self._looks_like_raw_dump(text):
                        return text
            except Exception as e:
                print(f"AI generation error: {e}")

        return self._fallback_response(results, query_metadata)

    def _fallback_response(self, results: dict, query_metadata: dict = None) -> str:
        structured_fraud_breakdown = self._format_fraud_category_breakdown(results)
        if structured_fraud_breakdown:
            response = structured_fraud_breakdown
            context = self._format_user_context(query_metadata)
            if context:
                response = f"{response}\n{context}"
            return response

        parts = []

        for tool_name, data in results.items():
            if not isinstance(data, dict) or "error" in data:
                continue

            if "fraud_count" in data and "fraud_rate" in data:
                total_transactions = int(data.get("total_transactions", 0))
                fraud_count = int(data.get("fraud_count", 0))
                fraud_rate = float(data.get("fraud_rate", 0))
                parts.append(
                    f"There are {fraud_count} flagged transactions out of {total_transactions}, "
                    f"for an overall fraud rate of {self._format_percent(fraud_rate)}."
                )
                top_clients = data.get("top_fraudulent_clients", [])[:3]
                if top_clients:
                    top_summary = ", ".join(
                        f"client {item['client_id']} at {self._format_percent(float(item['fraud_rate']))}"
                        for item in top_clients
                    )
                    parts.append(f"The highest observed fraud rates were {top_summary}.")
            elif "data" in data and "meta" in data:
                summary = self._summarize_table_result(data)
                if summary:
                    parts.append(summary)
            elif "risk_score" in data:
                parts.append(f"Fraud risk score: {data.get('risk_score')} ({data.get('risk_level', 'unknown')}).")
            elif "fraud_probability" in data:
                parts.append(f"Fraud probability: {self._format_percent(float(data.get('fraud_probability', 0)))}.")
            elif "credit_score" in data:
                parts.append(f"Credit score: {data.get('credit_score', 'N/A')}.")
            elif "total_spent" in data:
                parts.append(f"Total spending is ${float(data.get('total_spent', 0)):,.2f}.")
            elif "forecast" in data:
                forecast_values = ", ".join(
                    f"{month.replace('_', ' ')} ${float(value):,.2f}"
                    for month, value in list(data["forecast"].items())[:3]
                )
                sentence = f"Forecasted spending: {forecast_values}."
                confidence = data.get("confidence")
                total_months = data.get("total_months_data")
                latest_date = data.get("latest_transaction_date")
                warnings = set(data.get("warnings", []))

                if confidence == "low":
                    sentence += " This is a low-confidence estimate based on limited historical data."
                elif confidence == "medium":
                    sentence += " This is a rough estimate with moderate confidence."

                if total_months:
                    sentence += f" It uses {int(total_months)} historical spending months."
                if latest_date and "stale_history" in warnings:
                    sentence += f" The latest observed transaction is from {latest_date}, so the history is old."
                elif latest_date:
                    sentence += f" The latest observed transaction is from {latest_date}."

                if "sparse_month_coverage" in warnings:
                    sentence += " The observed spending months are spread out rather than continuous."

                parts.append(sentence)

        response = " ".join(parts) if parts else "I found some data."
        context = self._format_user_context(query_metadata)
        if context:
            response = f"{response} {context}"
        return response


ai_response_generator = AIResponseGenerator()


async def generate_ai_response(query: str, results: dict, query_type: str = "general") -> str:
    return await ai_response_generator.generate_response(query, results, query_type)
