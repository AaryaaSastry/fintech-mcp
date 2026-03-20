"""Test script for the cross-query conversation flow."""
import asyncio
import io
import sys

from backend.cross_query_handler import CrossQueryHandler, get_welcome_message
from backend.mcp_client import stop_mcp_client


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def test_cross_query():
    try:
        print("=" * 60)
        print("TESTING CROSS-QUERY CONVERSATION FLOW")
        print("=" * 60)

        print("\n[TEST 1] Welcome Message")
        print("-" * 40)
        welcome = get_welcome_message()
        print(welcome[:300] + "...")

        print("\n[TEST 2] Numeric Input")
        print("-" * 40)
        handler = CrossQueryHandler()
        result = await handler.process_query("2")
        print(f"Type: {result.get('type')}")
        if result.get("message"):
            print(f"Message preview: {result['message'][:200]}...")

        print("\n[TEST 3] Natural Language Query")
        print("-" * 40)
        handler2 = CrossQueryHandler()
        result2 = await handler2.process_query("show client 11 credit score history")
        print(f"Type: {result2.get('type')}")
        print(f"Query type detected: {handler2.last_query_type}")

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
    finally:
        await stop_mcp_client()


if __name__ == "__main__":
    asyncio.run(test_cross_query())
