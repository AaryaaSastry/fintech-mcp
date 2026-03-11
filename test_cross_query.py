"""
Test script for Cross-Query Conversation Flow
"""
import asyncio
import sys
import io
sys.path.insert(0, '.')

# Fix unicode for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from backend.cross_query_handler import CrossQueryHandler, get_welcome_message

async def test_cross_query():
    print("="*60)
    print("TESTING CROSS-QUERY CONVERSATION FLOW")
    print("="*60)
    
    # Test 1: Welcome message
    print("\n[TEST 1] Welcome Message with Menu")
    print("-"*40)
    welcome = get_welcome_message()
    print(welcome[:300] + "...")
    
    # Test 2: Menu selection
    print("\n[TEST 2] Menu Selection (2 = Credit Analysis)")
    print("-"*40)
    handler = CrossQueryHandler()
    result = await handler.process_query('2')
    print(f"Type: {result.get('type')}")
    print(f"Has graph suggestion: {'graph_suggestion' in result}")
    if result.get('message'):
        print(f"Message preview: {result['message'][:200]}...")
    
    # Test 3: Natural language query
    print("\n[TEST 3] Natural Language Query")
    print("-"*40)
    handler2 = CrossQueryHandler()
    result2 = await handler2.process_query('show james credit score history')
    print(f"Type: {result2.get('type')}")
    print(f"Query type detected: {handler2.last_query_type}")
    
    # Test 4: Yes for graph
    print("\n[TEST 4] YES for Graph Generation")
    print("-"*40)
    result3 = await handler2.process_query('yes')
    print(f"Type: {result3.get('type')}")
    print(f"Message: {result3.get('message', '')[:100]}...")
    
    # Test 5: No for graph
    print("\n[TEST 5] NO - Skip Graph")
    print("-"*40)
    handler3 = CrossQueryHandler()
    await handler3.process_query('1')
    result4 = await handler3.process_query('no')
    print(f"Type: {result4.get('type')}")
    print(f"Message: {result4.get('message')}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_cross_query())
