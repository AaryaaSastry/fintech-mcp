"""
MCP Orchestrator - Routes user queries to appropriate MCP tools
=============================================================
Supports 25+ analytical tools with LLM-based and keyword-based routing.
"""
import os
import json
import asyncio
import pandas as pd
from google import genai
from dotenv import load_dotenv
from .mcp_client import call_mcp_tool

load_dotenv()

# Configure Google Gemini
client = genai.Client()
MODEL_ID = "gemma-3-27b-it"  

# Client name mapping
CLIENT_NAME_MAP = {
    "soniya": 5, "sonia": 5, "john": 12, "emma": 15, "mike": 22, "sarah": 8
}

# All available tools metadata
TOOLS_METADATA = [
    # Single-client tools
    {"name": "get_spending_summary", "params": {"client_id": "int", "months": "int"}},
    {"name": "spending_by_category", "params": {"client_id": "int"}},
    {"name": "monthly_spending_trend", "params": {"client_id": "int"}},
    {"name": "spending_by_city", "params": {"client_id": "int"}},
    {"name": "spending_by_card_type", "params": {"client_id": "int"}},
    {"name": "total_transactions_count", "params": {"client_id": "int"}},
    {"name": "average_transaction_amount", "params": {"client_id": "int"}},
    {"name": "check_fraud", "params": {"transaction_id": "int"}},
    {"name": "client_fraud_risk", "params": {"client_id": "int"}},
    {"name": "fraud_rate_by_merchant", "params": {"mcc_code": "str"}},
    {"name": "high_risk_transactions", "params": {"client_id": "int", "threshold": "float"}},
    {"name": "credit_score_history", "params": {"client_id": "int"}},
    {"name": "debt_income_analysis", "params": {"client_id": "int"}},
    {"name": "credit_limit_analysis", "params": {"client_id": "int"}},
    {"name": "spending_forecast", "params": {"client_id": "int", "months": "int"}},
    {"name": "transaction_heatmap", "params": {"client_id": "int"}},
    {"name": "peak_hours_analysis", "params": {"client_id": "int"}},
    {"name": "spending_distribution", "params": {"client_id": "int"}},
    {"name": "merchant_summary", "params": {"mcc_code": "str"}},
    {"name": "merchant_risk_analysis", "params": {"mcc_code": "str"}},
    {"name": "top_merchants", "params": {"client_id": "int", "limit": "int"}},
    {"name": "card_usage_analysis", "params": {"client_id": "int"}},
    {"name": "card_type_spending", "params": {"client_id": "int"}},
    {"name": "customer_profile", "params": {"client_id": "int"}},
    {"name": "income_analysis", "params": {"client_id": "int"}},
    # ALL-USERS aggregate tools
    {"name": "get_all_users_spending_summary", "params": {"months": "int"}},
    {"name": "spending_by_category_all", "params": {}},
    {"name": "monthly_spending_trend_all", "params": {}},
    {"name": "spending_by_city_all", "params": {}},
    {"name": "spending_by_card_type_all", "params": {}},
    {"name": "total_transactions_all", "params": {}},
    {"name": "average_transaction_all", "params": {}},
    {"name": "global_fraud_analysis", "params": {}},
    {"name": "all_clients_fraud_summary", "params": {}},
    {"name": "credit_score_distribution_all", "params": {}},
    {"name": "debt_income_all", "params": {}},
    {"name": "credit_limit_all", "params": {}},
    {"name": "transaction_heatmap_all", "params": {}},
    {"name": "peak_hours_all", "params": {}},
    {"name": "spending_distribution_all", "params": {}},
    {"name": "merchant_summary_all", "params": {}},
    {"name": "top_merchants_all", "params": {"limit": "int"}},
    {"name": "card_usage_all", "params": {}},
    {"name": "customer_segmentation_all", "params": {}},
    {"name": "spending_forecast_all", "params": {"months": "int"}},
]

def get_transaction_for_client(client_id: int) -> int:
    """Get latest transaction_id for a client."""
    try:
        df = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip')
        client_txs = df[df['client_id'] == client_id]
        if not client_txs.empty:
            return int(client_txs['transaction_id'].iloc[-1])
    except: pass
    return None  # Return None instead of client_id to indicate not found

def client_exists(client_id: int) -> bool:
    """Check if a client exists in the transaction data."""
    try:
        df = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip')
        return client_id in df['client_id'].values
    except: 
        return False

def get_valid_client_id() -> int:
    """Get a valid default client_id from the data."""
    try:
        df = pd.read_csv("data/transactions.csv", engine='python', on_bad_lines='skip')
        if not df.empty:
            return int(df['client_id'].iloc[0])
    except: pass
    return 5  # Fallback to client 5

def fix_param_types(params: dict) -> dict:
    """Convert string params to proper types."""
    fixed = {}
    int_params = ["client_id", "transaction_id", "months", "threshold", "limit"]
    
    for k, v in params.items():
        if k in int_params:
            if isinstance(v, str):
                try:
                    fixed[k] = int(v)
                except:
                    fixed[k] = v
            else:
                fixed[k] = v
        else:
            fixed[k] = v
    return fixed

class MCPOrchestrator:
    def __init__(self):
        self.tools_metadata = TOOLS_METADATA

    def _keyword_fallback(self, query: str) -> list:
        """Keyword-based routing with ALL-USERS support."""
        q = query.lower()
        plan = []
        
        # Check if query is for ALL users (aggregate analysis)
        is_all_users = any(phrase in q for phrase in [
            "all users", "all clients", "everyone", "overall", 
            "global", "total", "aggregate", "across all", 
            "all the users", "all customers", "overall"
        ])
        
        # Extract client_id only if NOT all users
        client_id = None
        if not is_all_users:
            for name, cid in CLIENT_NAME_MAP.items():
                if name in q:
                    client_id = cid
                    break
            
            if not client_id:
                import re
                nums = re.findall(r'\d+', query)
                if nums:
                    client_id = int(nums[0])
            
            # Validate client_id - if not found, use a valid default
            if not client_id or not client_exists(client_id):
                client_id = get_valid_client_id()
        
        transaction_id = get_transaction_for_client(client_id) if client_id else None
        
        # ==================== ALL-USERS AGGREGATE ROUTING ====================
        if is_all_users:
            if any(w in q for w in ["spend", "summary", "total"]):
                plan.append({"tool": "get_all_users_spending_summary", "params": {"months": 3}})
            
            if any(w in q for w in ["category", "merchant", "pie"]):
                plan.append({"tool": "spending_by_category_all", "params": {}})
            
            if any(w in q for w in ["trend", "month"]):
                plan.append({"tool": "monthly_spending_trend_all", "params": {}})
            
            if any(w in q for w in ["city", "location"]):
                plan.append({"tool": "spending_by_city_all", "params": {}})
            
            if any(w in q for w in ["card"]):
                plan.append({"tool": "spending_by_card_type_all", "params": {}})
            
            if "count" in q:
                plan.append({"tool": "total_transactions_all", "params": {}})
            
            if "average" in q:
                plan.append({"tool": "average_transaction_all", "params": {}})
            
            if any(w in q for w in ["fraud", "risky", "risk"]):
                plan.append({"tool": "global_fraud_analysis", "params": {}})
                plan.append({"tool": "all_clients_fraud_summary", "params": {}})
            
            if "forecast" in q or "future" in q:
                plan.append({"tool": "spending_forecast_all", "params": {"months": 3}})
            
            if any(w in q for w in ["heatmap", "day", "hour"]):
                plan.append({"tool": "transaction_heatmap_all", "params": {}})
            
            if "peak" in q:
                plan.append({"tool": "peak_hours_all", "params": {}})
            
            if "distribution" in q:
                plan.append({"tool": "spending_distribution_all", "params": {}})
            
            if "debt" in q or "income" in q:
                plan.append({"tool": "debt_income_all", "params": {}})
            
            if "credit score" in q:
                plan.append({"tool": "credit_score_distribution_all", "params": {}})
            
            if "credit limit" in q:
                plan.append({"tool": "credit_limit_all", "params": {}})
            
            if "segment" in q:
                plan.append({"tool": "customer_segmentation_all", "params": {}})
            
            if "card usage" in q or "chip" in q:
                plan.append({"tool": "card_usage_all", "params": {}})
            
            if "top merchant" in q:
                plan.append({"tool": "top_merchants_all", "params": {"limit": 10}})
            
            if "merchant" in q:
                plan.append({"tool": "merchant_summary_all", "params": {}})
            
            return plan
        
        # ==================== SINGLE-USER ROUTING ====================
        # Keyword routing
        if any(w in q for w in ["spend", "summary", "total"]):
            plan.append({"tool": "get_spending_summary", "params": {"client_id": client_id, "months": 3}})
        
        if any(w in q for w in ["category", "merchant", "pie"]):
            plan.append({"tool": "spending_by_category", "params": {"client_id": client_id}})
        
        if any(w in q for w in ["trend", "month"]):
            plan.append({"tool": "monthly_spending_trend", "params": {"client_id": client_id}})
        
        if any(w in q for w in ["city", "location"]):
            plan.append({"tool": "spending_by_city", "params": {"client_id": client_id}})
        
        if any(w in q for w in ["card"]):
            plan.append({"tool": "spending_by_card_type", "params": {"client_id": client_id}})
        
        if "count" in q:
            plan.append({"tool": "total_transactions_count", "params": {"client_id": client_id}})
        
        if "average" in q:
            plan.append({"tool": "average_transaction_amount", "params": {"client_id": client_id}})
        
        if any(w in q for w in ["fraud", "risky", "risk"]):
            # Only add check_fraud if we have a valid transaction_id
            if transaction_id is not None:
                plan.append({"tool": "check_fraud", "params": {"transaction_id": transaction_id}})
            # Always add client_fraud_risk - it will handle the error if client doesn't exist
            plan.append({"tool": "client_fraud_risk", "params": {"client_id": client_id}})
        
        if "forecast" in q or "future" in q:
            plan.append({"tool": "spending_forecast", "params": {"client_id": client_id, "months": 3}})
        
        if any(w in q for w in ["heatmap", "day", "hour"]):
            plan.append({"tool": "transaction_heatmap", "params": {"client_id": client_id}})
        
        if "peak" in q:
            plan.append({"tool": "peak_hours_analysis", "params": {"client_id": client_id}})
        
        if "distribution" in q:
            plan.append({"tool": "spending_distribution", "params": {"client_id": client_id}})
        
        if "debt" in q or "income" in q:
            plan.append({"tool": "debt_income_analysis", "params": {"client_id": client_id}})
        
        if "credit score" in q:
            plan.append({"tool": "credit_score_history", "params": {"client_id": client_id}})
        
        if "credit limit" in q:
            plan.append({"tool": "credit_limit_analysis", "params": {"client_id": client_id}})
        
        if "profile" in q:
            plan.append({"tool": "customer_profile", "params": {"client_id": client_id}})
        
        if "card usage" in q or "chip" in q:
            plan.append({"tool": "card_usage_analysis", "params": {"client_id": client_id}})
        
        if "top merchant" in q:
            plan.append({"tool": "top_merchants", "params": {"client_id": client_id, "limit": 5}})
        
        return plan

    async def handle_query(self, user_query: str):
        """Main query handler."""
        # Use keyword fallback only (skip LLM for faster results)
        plan = self._keyword_fallback(user_query)
        
        print(f"Plan: {plan}")
        
        # Execute tools (direct calls - much faster than MCP)
        results = {}
        for step in plan:
            t_name = step.get("tool")
            params = fix_param_types(step.get("params", {}))
            print(f"Executing: {t_name} with {params}")
            
            try:
                # Use MCP client for tool execution (persistent session - faster)
                res = await call_mcp_tool(t_name, params)
                results[t_name] = res
            except Exception as e:
                results[t_name] = {"error": str(e)}

        if not results:
            return {"summary": "No data found."}
        
        # Generate summary (skip LLM for faster results - just format results)
        try:
            # Build a simple summary from results
            summary_parts = []
            for tool_name, result in results.items():
                if isinstance(result, dict) and "error" not in result:
                    if "risk_score" in result:
                        summary_parts.append(f"{tool_name}: Risk score {result.get('risk_score')} ({result.get('risk_level')})")
                    elif "fraud_probability" in result:
                        summary_parts.append(f"{tool_name}: Fraud probability {result.get('fraud_probability')}")
                    else:
                        summary_parts.append(f"{tool_name}: {len(result)} items")
                elif isinstance(result, dict) and "error" in result:
                    summary_parts.append(f"{tool_name}: {result.get('error')}")
            
            summary = "; ".join(summary_parts) if summary_parts else "Results available"
        except:
            summary = " | ".join([f"{k}" for k in results.keys()])
        
        return {"summary": summary, "raw_data": results, "plan": plan}

orchestrator = MCPOrchestrator()
