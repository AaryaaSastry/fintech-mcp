import asyncio
import importlib
import json
from typing import Any, Dict

# Direct in-process dispatch to MCP tool functions defined in mcp_server/server.py
_server_module = None


def _get_tool_func(tool_name: str):
    """Lazy-load the server module and fetch the tool function by name."""
    global _server_module
    if _server_module is None:
        _server_module = importlib.import_module("mcp_server.server")
    return getattr(_server_module, tool_name, None)


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Execute a tool in-process (no subprocess/stdout round-trip).
    Keeps behavior identical to MCP tools while eliminating startup latency.
    """
    func = _get_tool_func(tool_name)
    if func is None:
        return {"error": f"Tool '{tool_name}' not found"}

    loop = asyncio.get_running_loop()
    try:
        # Run the sync tool function in a thread to avoid blocking the event loop.
        result = await loop.run_in_executor(None, lambda: func(**arguments))
        return result
    except Exception as e:
        return {"error": f"MCP Client Error: {e}"}


if __name__ == "__main__":
    async def test():
        res = await call_mcp_tool("get_spending_summary", {"client_id": 5, "months": 3})
        print(json.dumps(res, indent=2))

    asyncio.run(test())
