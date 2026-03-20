"""
Cross-Query Conversation CLI
=============================
Run the conversational AI from your terminal.
"""
import asyncio
import io
import sys

from backend.cross_query_handler import CrossQueryHandler, get_welcome_message
from backend.mcp_client import stop_mcp_client


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


async def main():
    """Main conversation loop."""
    try:
        print("\n" + "=" * 60)
        print("  FINTECH ASSISTANT")
        print("=" * 60 + "\n")

        handler = CrossQueryHandler()
        print(get_welcome_message())
        print("\n" + "-" * 50)

        while True:
            try:
                user_input = input("\n>>> You: ").strip()
                if not user_input:
                    continue

                result = await handler.process_query(user_input)
                if result.get("type") == "exit":
                    print(f"\n{result.get('message')}")
                    break

                if result.get("message"):
                    print(f"\n{result.get('message')}")
            except KeyboardInterrupt:
                print("\n\nTake care. Goodbye.")
                break
            except Exception as exc:
                print(f"\nOops, something went wrong: {exc}")
                print("Let's try again.")
    finally:
        await stop_mcp_client()


if __name__ == "__main__":
    asyncio.run(main())
