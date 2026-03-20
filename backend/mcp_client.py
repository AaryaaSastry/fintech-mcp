import json
import os
import sys
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


_session: ClientSession | None = None
_exit_stack: AsyncExitStack | None = None
_session_lock: asyncio.Lock | None = None
_call_lock: asyncio.Lock | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _server_params() -> StdioServerParameters:
    root = _project_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(root),
        env=env,
    )


async def start_mcp_client() -> None:
    global _session, _exit_stack, _session_lock, _call_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    if _call_lock is None:
        _call_lock = asyncio.Lock()

    assert _session_lock is not None
    async with _session_lock:
        if _session is not None:
            return

        stack = AsyncExitStack()
        read_stream, write_stream = await stack.enter_async_context(stdio_client(_server_params()))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        _exit_stack = stack
        _session = session


async def stop_mcp_client() -> None:
    global _session, _exit_stack, _session_lock
    if _session_lock is None:
        _session_lock = asyncio.Lock()

    assert _session_lock is not None
    async with _session_lock:
        if _exit_stack is not None:
            await _exit_stack.aclose()
        _session = None
        _exit_stack = None


def _normalize_call_result(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        if len(content) == 1:
            item = content[0]
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text

        normalized = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    normalized.append(json.loads(text))
                except json.JSONDecodeError:
                    normalized.append(text)
            else:
                normalized.append(item.model_dump() if hasattr(item, "model_dump") else str(item))
        return normalized

    return result.model_dump() if hasattr(result, "model_dump") else result


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    if _session is None:
        await start_mcp_client()

    assert _session is not None
    global _call_lock
    if _call_lock is None:
        _call_lock = asyncio.Lock()

    assert _call_lock is not None
    async with _call_lock:
        try:
            result = await _session.call_tool(tool_name, arguments)
            normalized = _normalize_call_result(result)
            if getattr(result, "isError", False) and isinstance(normalized, dict) and "error" not in normalized:
                return {"error": normalized}
            return normalized
        except Exception as e:
            return {"error": f"MCP Client Error: {e}"}
