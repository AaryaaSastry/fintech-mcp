"""
AI Response Generator
====================
Uses LLM to generate natural, conversational responses based on financial data outputs.
"""
import json
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configure Google Gemini
client = genai.Client()
MODEL_ID = "gemma-3-27b-it"


class AIResponseGenerator:
    """
    Generates AI-powered conversational responses based on query and tool results.
    """
    
    def __init__(self):
        self.model_id = MODEL_ID
    
    def _build_prompt(self, query: str, results: dict, query_type: str) -> str:
        """
        Build a prompt for the LLM to generate a natural response.
        """
        # Format the results for the prompt
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
        
        prompt = f"""You are a friendly fintech assistant having a conversation with a user.
Your goal is to explain financial data in a natural, conversational way - like telling a friend about their finances.

USER'S QUESTION: {query}
QUERY TYPE: {query_type}

TOOL RESULTS:
{results_str}

INSTRUCTIONS:
1. Start with a friendly conversational opener that relates to what they asked
2. Explain the data in simple, easy-to-understand terms
3. Use natural language - avoid technical jargon
4. Be specific with numbers and percentages when available
5. Add helpful insights or context when relevant
6. End with a casual follow-up question or offer to help more
7. Keep it conversational and friendly - not formal or robotic

Remember: You're talking to a human, not presenting data. Make it feel like a helpful friend explaining their finances.
"""
        return prompt
    
    async def generate_response(self, query: str, results: dict, query_type: str = "general") -> str:
        """
        Generate an AI response based on the query and tool results.
        """
        if not results:
            return "I couldn't find any data for that. Want to try a different question?"
        
        # Check if there are errors in results
        has_errors = all(
            isinstance(v, dict) and "error" in v 
            for v in results.values()
        ) if results else True
        
        if has_errors:
            return "I ran into some issues getting that information. Would you like me to try again or ask about something else?"
        
        try:
            prompt = self._build_prompt(query, results, query_type)
            
            response = client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            if response.text:
                return response.text.strip()
            else:
                # Fallback to template-based response if LLM fails
                return self._fallback_response(query, results, query_type)
                
        except Exception as e:
            print(f"AI generation error: {e}")
            # Fallback to template-based response
            return self._fallback_response(query, results, query_type)
    
    def _fallback_response(self, query: str, results: dict, query_type: str) -> str:
        """
        Fallback template-based response if AI generation fails.
        """
        parts = []
        
        for tool_name, data in results.items():
            if isinstance(data, dict) and "error" not in data:
                if "risk_score" in data:
                    risk = data.get("risk_score", 0)
                    level = data.get("risk_level", "Unknown")
                    parts.append(f"🛡️ Your fraud risk score is {risk} out of 100 ({level} risk).")
                
                elif "fraud_probability" in data:
                    prob = data.get("fraud_probability", 0)
                    parts.append(f"⚠️ Fraud probability: {prob}%")
                
                elif "credit_score" in data:
                    cs = data.get("credit_score", "N/A")
                    parts.append(f"📈 Your credit score is {cs}.")
                    if "credit_limit" in data:
                        limit = data.get('credit_limit')
                        parts.append(f"Credit limit: ${limit:,.2f}")
                
                elif "total_spent" in data:
                    total = data.get("total_spent", 0)
                    parts.append(f"💰 Total spending: ${total:,.2f}")
                    if "transaction_count" in data:
                        count = data.get("transaction_count", 0)
                        parts.append(f"Across {count} transactions.")
                
                elif "predicted_spending" in data:
                    pred = data.get("predicted_spending", 0)
                    parts.append(f"🔮 Predicted spending: ${pred:,.2f}")
                
                else:
                    # Generic display
                    items = []
                    for k, v in list(data.items())[:3]:
                        if not k.startswith("_"):
                            items.append(f"{k}: {v}")
                    if items:
                        parts.append(f"📌 " + ", ".join(items))
        
        response = " ".join(parts) if parts else "I found some data. Would you like more details?"
        
        # Add casual follow-up
        follow_ups = [
            "\n\nAnything else you'd like to know?",
            "\n\nLet me know if you need anything else!",
            "\n\nFeel free to ask more questions!",
        ]
        import random
        response += random.choice(follow_ups)
        
        return response


# Global instance
ai_response_generator = AIResponseGenerator()


async def generate_ai_response(query: str, results: dict, query_type: str = "general") -> str:
    """
    Convenience function to generate AI response.
    """
    return await ai_response_generator.generate_response(query, results, query_type)
