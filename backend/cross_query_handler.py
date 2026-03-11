"""
Cross-Query Conversation Handler
=================================
A conversational AI interface that:
1. Engages in natural dialogue about financial data
2. Responds conversationally without menus or structured prompts
3. Provides information in a friendly, human-like manner
4. Loops for continuous interaction
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum

# Import the orchestrator for query processing
from backend.mcp_orchestrator import orchestrator
from backend.mcp_client import call_mcp_tool
from backend.ai_response_generator import ai_response_generator


class ConversationState(Enum):
    """States in the conversation flow."""
    GREETING = "greeting"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"


# Helpful responses for different situations
GREETING_RESPONSES = [
    "Hey there! I'm here to help you with your finances. What would you like to know?",
    "Hi! I'm your fintech assistant. Ask me anything about your spending, credit, or finances!",
    "Hello! I can help you understand your financial data. Just tell me what you need!",
]

HELP_OPTIONS = """
Here are some things I can help you with:

• How much did I spend this month?
• What's my credit score?
• Are there any suspicious transactions?
• How much will I spend next month?
• Show me my spending by category
• What's my spending trend?

Just ask in your own words!
"""

FAREWELL_RESPONSES = [
    "Take care! It was great chatting with you!",
    "Goodbye! Feel free to come back anytime!",
    "Bye! Thanks for stopping by!",
]


class CrossQueryHandler:
    """
    Handles the cross-query conversational flow in a natural, friendly way.
    """
    
    def __init__(self):
        self.state = ConversationState.GREETING
        self.last_result = None
        self.last_query_type = None
        self.conversation_active = True
        self.greeting_count = 0
    
    def get_greeting(self) -> str:
        """Return a friendly greeting."""
        import random
        greeting = random.choice(GREETING_RESPONSES)
        self.greeting_count += 1
        return greeting
    
    def parse_input(self, user_input: str) -> str:
        """
        Parse user input and determine what to do.
        """
        user_input = user_input.strip()
        
        if not user_input:
            return "__EMPTY__"
        
        lower_input = user_input.lower()
        
        # Check for exit/quit
        if lower_input in ["exit", "quit", "bye", "goodbye", "see you", "later"]:
            return "__EXIT__"
        
        # Check for help
        if lower_input in ["help", "what can you do", "options", "menu", "commands"]:
            return "__HELP__"
        
        # Check for greetings
        if lower_input in ["hi", "hello", "hey", "hiya", "greetings"]:
            return "__GREETING__"
        
        # Check for thanks
        if lower_input in ["thanks", "thank you", "thx", "appreciate"]:
            return "__THANKS__"
        
        # Check for small talk
        if lower_input in ["how are you", "how do you do", "what's up", "whats up"]:
            return "__HOW_ARE_YOU__"
        
        # Otherwise, treat as a query
        return user_input
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query."""
        q = query.lower()
        
        if "credit" in q and "score" in q:
            return "credit_score"
        if ("spend" in q or "spent" in q) and ("forecast" in q or "predict" in q or "future" in q):
            return "forecast"
        if "fraud" in q or "risk" in q or "suspicious" in q:
            return "fraud"
        if "trend" in q or "month" in q:
            return "trend"
        if "category" in q or "categories" in q:
            return "category"
        if "spend" in q or "spent" in q or "total" in q:
            return "spending"
        
        return "general"
    
    def format_data_conversationally(self, result: Dict[str, Any]) -> str:
        """
        Format the query result in a conversational way - like explaining to a friend.
        """
        if not result:
            return "I couldn't find any data for that. Want to try a different question?"
        
        summary = result.get("summary", "") if isinstance(result, dict) else ""
        raw_data = result.get("raw_data", {}) if isinstance(result, dict) else {}
        
        if not isinstance(raw_data, dict):
            return "I received data but couldn't interpret it properly."
        
        # Build conversational output
        parts = []
        
        # Process each tool result
        for tool_name, data in raw_data.items():
            if isinstance(data, dict):
                if "error" in data:
                    parts.append(f"Oops, there was an issue: {data.get('error')}")
                    continue
                
                # Format based on data type - conversational style
                if "risk_score" in data:
                    risk = data.get("risk_score", 0)
                    level = data.get("risk_level", "Unknown")
                    parts.append(f"🛡️ Based on my analysis, your fraud risk score is {risk} out of 100 ({level} risk).")
                    if "factors" in data:
                        factors = data.get('factors', [])
                        if factors:
                            parts.append(f"The main factors are: {', '.join(factors)}.")
                
                elif "fraud_probability" in data:
                    prob = data.get("fraud_probability", 0)
                    parts.append(f"⚠️ The probability of fraud is {prob}%. Pretty low, so you're looking good!")
                
                elif "credit_score" in data:
                    cs = data.get("credit_score", "N/A")
                    parts.append(f"📈 Your credit score is {cs}.")
                    if "history" in data:
                        parts.append(f"According to your history: {data.get('history')}.")
                    if "debt_income_ratio" in data:
                        ratio = data.get('debt_income_ratio')
                        parts.append(f"Your debt-to-income ratio is {ratio}.")
                    if "credit_limit" in data:
                        limit = data.get('credit_limit')
                        parts.append(f"And you've got a credit limit of ${limit:,.2f}.")
                
                elif "total_spent" in data:
                    total = data.get("total_spent", 0)
                    parts.append(f"💰 You've spent a total of ${total:,.2f}.")
                    if "avg_transaction" in data:
                        avg = data.get("avg_transaction", 0)
                        parts.append(f"Each transaction averages around ${avg:,.2f}.")
                    if "transaction_count" in data:
                        count = data.get("transaction_count", 0)
                        parts.append(f"That's across {count} transactions.")
                
                elif "forecast" in tool_name or "predicted" in str(data):
                    parts.append(f"🔮 Here's what I'm predicting:")
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            parts.append(f"  • {k}: ${v:,.2f}")
                
                else:
                    # Generic conversational display
                    items = []
                    for k, v in list(data.items())[:4]:
                        if not k.startswith("_"):
                            items.append(f"{k}: {v}")
                    if items:
                        parts.append(f"📌 Here's what I found: {', '.join(items)}")
            
            elif isinstance(data, list):
                if data:
                    items = []
                    for item in data[:5]:
                        items.append(str(item))
                    if items:
                        parts.append(f"📋 {tool_name.replace('_', ' ').title()}: {', '.join(items)}")
        
        # Add conversational summary
        if summary:
            parts.append(f"\n💡 In short: {summary}")
        
        if not parts:
            parts.append("I found some data but wasn't sure how to present it. Try asking in a different way?")
        
        return " ".join(parts)
    
    async def process_query(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and return conversational response.
        """
        # Parse the input
        query = self.parse_input(user_input)
        
        # Handle special commands conversationally
        if query == "__EXIT__":
            import random
            return {
                "message": random.choice(FAREWELL_RESPONSES),
                "type": "exit"
            }
        
        if query == "__HELP__":
            return {
                "message": HELP_OPTIONS,
                "type": "help"
            }
        
        if query == "__GREETING__":
            return {
                "message": self.get_greeting(),
                "type": "greeting"
            }
        
        if query == "__EMPTY__":
            return {
                "message": "I didn't catch that. What would you like to know?",
                "type": "empty"
            }
        
        if query == "__THANKS__":
            return {
                "message": "You're welcome! Happy to help! What else would you like to know?",
                "type": "thanks"
            }
        
        if query == "__HOW_ARE_YOU__":
            return {
                "message": "I'm doing great, thanks for asking! Ready to help you with your finances. What do you need?",
                "type": "small_talk"
            }
        
        # Process actual query
        if query:
            self.state = ConversationState.PROCESSING
            
            # Detect query type
            self.last_query_type = self.detect_query_type(query)
            
            # Process through orchestrator
            try:
                result = await orchestrator.handle_query(query)
                print("DEBUG RESULT:", result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "message": f"⚠️ Something went wrong in the financial engine: {str(e)}",
                    "type": "error"
                }
            
            if not result:
                return {
                    "message": "I couldn't retrieve any financial data for that query.",
                    "type": "error"
                }
            
            self.last_result = result
            
            # Format conversational response using AI
            conversational_output = await ai_response_generator.generate_response(
                query, 
                result.get("raw_data", {}), 
                self.last_query_type
            )
            
            # Add a conversational follow-up
            follow_ups = [
                "\n\nAnything else you'd like to know?",
                "\n\nFeel free to ask more questions!",
                "\n\nLet me know if you need anything else!",
                "\n\nWant me to look into anything else?",
            ]
            import random
            follow_up = random.choice(follow_ups)
            
            return {
                "message": conversational_output + follow_up,
                "type": "result"
            }
        
        return {
            "message": "I'm not sure I understood that. Could you try rephrasing?",
            "type": "error"
        }


# Global handler instance
cross_query_handler = CrossQueryHandler()


async def handle_conversation(user_input: str) -> Dict[str, Any]:
    """
    Main entry point for handling conversational input.
    """
    return await cross_query_handler.process_query(user_input)


def get_welcome_message() -> str:
    """
    Get the welcome message - completely conversational style.
    """
    return """
👋 Hey there! I'm your fintech assistant.

I can help you with:
• Your spending habits and trends
• Credit score analysis  
• Fraud risk checks
• Future spending forecasts

Just tell me what you'd like to know! For example:
  "How much did I spend last month?"
  "What's my credit score?"
  "Are there any suspicious transactions?"

Or say "help" to see all the things I can do.
"""
