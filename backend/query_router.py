from fastapi import APIRouter, HTTPException, Query

try:
    from .mcp_client import call_mcp_tool
    from .schemas import QueryRequest
    from .mcp_orchestrator import orchestrator
    from .cross_query_handler import handle_conversation, get_welcome_message
except ImportError:
    from mcp_client import call_mcp_tool
    from schemas import QueryRequest
    from mcp_orchestrator import orchestrator
    from cross_query_handler import handle_conversation, get_welcome_message

router = APIRouter()

# Conversation session endpoint
@router.get("/conversation/start")
async def start_conversation():
    """Start a new conversation with the menu."""
    return {
        "message": get_welcome_message(),
        "type": "menu"
    }

@router.post("/conversation/chat")
async def chat_with_conversation(request: dict = None):
    """Chat endpoint that supports the full conversational flow."""
    if request is None:
        raise HTTPException(status_code=400, detail="Request body is required")
    
    user_input = None
    if isinstance(request, dict):
        user_input = request.get("message") or request.get("query") or request.get("text")
    
    if not user_input:
        raise HTTPException(status_code=400, detail="message field is required")
    
    print(f"Conversation input: {user_input}")
    
    result = await handle_conversation(user_input)
    return result

# Also support GET requests for testing
@router.get("/chat")
async def chat_get(q: str = Query(..., description="The query string")):
    """GET endpoint for testing - use ?q=your query"""
    print(f"GET received query: {q}")
    result = await orchestrator.handle_query(q)
    return result

@router.post("/chat")
async def chat_with_fintech_ai(request: dict = None):
    # Handle both JSON body and form data
    if request is None:
        raise HTTPException(status_code=400, detail="Request body is required")
    
    # Extract query from request
    user_query = None
    if isinstance(request, dict):
        user_query = request.get("query") or request.get("message") or request.get("text")
    
    if not user_query:
        raise HTTPException(status_code=400, detail="query field is required")
    
    print(f"Received query: {user_query}")
    
    # Use the Google Gemini Orchestrator instead of basic parsing
    result = await orchestrator.handle_query(user_query)
    
    # Handle errors from orchestrator gracefully
    if isinstance(result, dict) and "error" in result:
        # Return the error in a user-friendly format
        return {
            "summary": result.get("summary", "An error occurred"),
            "error": result.get("error"),
            "raw_data": result.get("raw_data", {}),
            "plan": result.get("plan", [])
        }
        
    return result

@router.post("/analytics/spending")
async def get_spending(request: QueryRequest):
    if not request.client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    return await call_mcp_tool("get_spending_summary", {"client_id": request.client_id, "months": request.months})

@router.post("/analytics/fraud-check")
async def check_fraud(request: QueryRequest):
    if not request.transaction_id:
        raise HTTPException(status_code=400, detail="transaction_id is required")
    return await call_mcp_tool("check_fraud", {"transaction_id": request.transaction_id})

@router.post("/analytics/merchant")
async def get_merchant_risk(request: QueryRequest):
    if not request.mcc_code:
        raise HTTPException(status_code=400, detail="mcc_code is required")
    return await call_mcp_tool("merchant_summary", {"mcc_code": request.mcc_code})

@router.post("/analytics/forecast")
async def get_forecast(request: QueryRequest):
    if not request.client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    return await call_mcp_tool("spending_forecast", {"client_id": request.client_id, "months": request.months})

@router.post("/analytics/heatmap")
async def get_heatmap(request: QueryRequest):
    if not request.client_id:
        raise HTTPException(status_code=400, detail="client_id is required")
    return await call_mcp_tool("transaction_heatmap", {"client_id": request.client_id})
