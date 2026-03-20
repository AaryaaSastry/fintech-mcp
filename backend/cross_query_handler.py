"""Conversational wrapper around the orchestrator and response generator."""
import random
from typing import Any, Dict

try:
    from backend.ai_response_generator import ai_response_generator
    from backend.mcp_orchestrator import orchestrator
except ImportError:
    from ai_response_generator import ai_response_generator
    from mcp_orchestrator import orchestrator


GREETING_RESPONSES = [
    "Hey there! I'm here to help you with your finances. What would you like to know?",
    "Hi! I'm your fintech assistant. Ask me anything about your spending, credit, or finances!",
    "Hello! I can help you understand your financial data. Just tell me what you need!",
]

HELP_OPTIONS = """
Here are some things I can help you with:

- How much did I spend this month?
- What's my credit score?
- Are there any suspicious transactions?
- How much will I spend next month?
- Show me my spending by category
- What's my spending trend?

Ask in your own words.
"""

FAREWELL_RESPONSES = [
    "Take care! It was great chatting with you!",
    "Goodbye! Feel free to come back anytime!",
    "Bye! Thanks for stopping by!",
]

FOLLOW_UPS = [
    "\n\nAnything else you'd like to know?",
    "\n\nFeel free to ask more questions!",
    "\n\nLet me know if you need anything else!",
    "\n\nWant me to look into anything else?",
]


class CrossQueryHandler:
    """Handles lightweight conversational commands plus analytics queries."""

    def __init__(self):
        self.pending_clarification: dict | None = None

    def parse_input(self, user_input: str) -> str:
        cleaned = user_input.strip()
        if not cleaned:
            return "__EMPTY__"

        lower_input = cleaned.lower()
        if lower_input in {"exit", "quit", "bye", "goodbye", "see you", "later"}:
            return "__EXIT__"
        if lower_input in {"help", "what can you do", "options", "menu", "commands"}:
            return "__HELP__"
        if lower_input in {"hi", "hello", "hey", "hiya", "greetings"}:
            return "__GREETING__"
        if lower_input in {"thanks", "thank you", "thx", "appreciate"}:
            return "__THANKS__"
        if lower_input in {"how are you", "how do you do", "what's up", "whats up"}:
            return "__HOW_ARE_YOU__"
        return cleaned

    def detect_query_type(self, query: str) -> str:
        q = query.lower()
        if "credit" in q and "score" in q:
            return "credit_score"
        if ("spend" in q or "spent" in q) and any(word in q for word in ["forecast", "forcast", "predict", "future"]):
            return "forecast"
        if any(word in q for word in ["fraud", "risk", "suspicious"]):
            return "fraud"
        if "trend" in q or "month" in q or "history" in q:
            return "trend"
        if "category" in q or "categories" in q:
            return "category"
        if any(word in q for word in ["spend", "spent", "total"]):
            return "spending"
        return "general"

    async def process_query(self, user_input: str) -> Dict[str, Any]:
        query = self.parse_input(user_input)

        if query == "__EXIT__":
            return {"message": random.choice(FAREWELL_RESPONSES), "type": "exit"}
        if query == "__HELP__":
            return {"message": HELP_OPTIONS, "type": "help"}
        if query == "__GREETING__":
            return {"message": random.choice(GREETING_RESPONSES), "type": "greeting"}
        if query == "__EMPTY__":
            return {"message": "I didn't catch that. What would you like to know?", "type": "empty"}
        if query == "__THANKS__":
            return {"message": "You're welcome. What else would you like to know?", "type": "thanks"}
        if query == "__HOW_ARE_YOU__":
            return {
                "message": "I'm ready to help with your finances. What do you need?",
                "type": "small_talk",
            }

        if self.pending_clarification is not None:
            selected = query.strip()
            option_map = {option["id"]: option for option in self.pending_clarification["options"]}
            if selected in option_map:
                query = option_map[selected]["query"]
                self.pending_clarification = None
            else:
                self.pending_clarification = None

        query_type = self.detect_query_type(query)
        try:
            result = await orchestrator.handle_query(query)
            print("DEBUG RESULT:", result)
        except Exception as exc:
            return {
                "message": f"Something went wrong in the financial engine: {exc}",
                "type": "error",
            }

        if not result:
            return {
                "message": "I couldn't retrieve any financial data for that query.",
                "type": "error",
            }

        if result.get("type") == "clarification":
            clarification = result["clarification"]
            if clarification.get("options"):
                self.pending_clarification = clarification
                options_text = "\n".join(
                    f"{option['id']}. {option['label']}" for option in clarification["options"]
                )
                return {
                    "message": (
                        f"{clarification['reason']}\n\n"
                        f"Here are the three closest interpretations of what I understood:\n"
                        f"{options_text}\n\n"
                        "Reply with 1, 2, or 3, and I'll run that."
                    ),
                    "type": "clarification",
                }

            return {
                "message": (
                    f"{clarification['reason']}\n\n"
                    f"{clarification.get('fallback_message', 'Please rephrase what you want me to show.')}"
                ),
                "type": "clarification",
            }

        conversational_output = await ai_response_generator.generate_response(
            query,
            result.get("raw_data", {}),
            query_type,
            result.get("query_metadata"),
            prefer_llm=result.get("planner_used", False),
        )
        return {
            "message": conversational_output + random.choice(FOLLOW_UPS),
            "type": "result",
        }


cross_query_handler = CrossQueryHandler()


async def handle_conversation(user_input: str) -> Dict[str, Any]:
    """Main entry point for handling conversational input."""
    return await cross_query_handler.process_query(user_input)


def get_welcome_message() -> str:
    """Get the conversational welcome message."""
    return """
Hey there! I'm your fintech assistant.

I can help you with:
- Your spending habits and trends
- Credit score analysis
- Fraud risk checks
- Future spending forecasts

Just tell me what you'd like to know. For example:
  "How much did I spend last month?"
  "What's my credit score?"
  "Are there any suspicious transactions?"

Or say "help" to see more examples.
"""
