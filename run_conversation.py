"""
Cross-Query Conversation CLI
=============================
Run the conversational AI from your terminal - entirely conversational!

Usage:
    python run_conversation.py
"""
import asyncio
import sys
import io

# Fix unicode for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from backend.cross_query_handler import CrossQueryHandler, get_welcome_message


async def main():
    """Main conversation loop - completely conversational."""
    print("\n" + "="*60)
    print("  FINTECH ASSISTANT")
    print("="*60 + "\n")
    
    # Initialize handler
    handler = CrossQueryHandler()
    
    # Show welcome message
    print(get_welcome_message())
    print("\n" + "-"*50)
    
    # Conversation loop
    while True:
        try:
            # Get user input
            user_input = input("\n>>> You: ").strip()
            
            if not user_input:
                continue
            
            # Process the input
            result = await handler.process_query(user_input)
            
            # Handle exit
            if result.get("type") == "exit":
                print(f"\n{result.get('message')}")
                break
            
            # Print the conversational response
            if result.get("message"):
                print(f"\n{result.get('message')}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Take care! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Oops, something went wrong: {e}")
            print("Let's try again!")


if __name__ == "__main__":
    asyncio.run(main())
