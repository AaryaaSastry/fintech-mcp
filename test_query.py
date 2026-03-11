#!/usr/bin/env python
"""CLI output for MCP queries."""
import asyncio
import sys
import json
from backend.mcp_orchestrator import orchestrator

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(data):
    """Pretty print the result."""
    if not data:
        print("No data returned")
        return
    
    # Print summary
    if "summary" in data:
        print_header("SUMMARY")
        print(f"\n{data['summary']}\n")
    
    # Print raw data
    if "raw_data" in data:
        print_header("DETAILED RESULTS")
        
        for tool_name, result in data["raw_data"].items():
            print(f"\n[{tool_name.replace('_', ' ').title()}]")
            print("-" * 40)
            
            if isinstance(result, dict):
                if "error" in result:
                    print(f"  Error: {result['error']}")
                else:
                    for key, value in result.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for k, v in value.items():
                                print(f"    - {k}: {v}")
                        elif isinstance(value, list):
                            print(f"  {key}:")
                            for item in value[:5]:
                                print(f"    - {item}")
                        else:
                            print(f"  {key}: {value}")
            else:
                print(f"  {result}")
    
    print("\n" + "="*60)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_query.py \"Your query here\"")
        print("Example: python test_query.py \"Show John spending\"")
        return
    
    query = " ".join(sys.argv[1:])
    
    print_header(f"QUERY: {query}")
    print("\n[Processing...]")
    
    result = await orchestrator.handle_query(query)
    print_result(result)

if __name__ == "__main__":
    asyncio.run(main())
