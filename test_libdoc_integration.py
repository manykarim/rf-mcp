#!/usr/bin/env python3
"""Test script to verify robot.libdoc integration in KeywordMatcher."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from robotmcp.components.keyword_matcher import KeywordMatcher

async def test_libdoc_integration():
    """Test the improved keyword discovery using robot.libdoc."""
    
    print("Testing robot.libdoc integration...")
    print("=" * 50)
    
    matcher = KeywordMatcher()
    
    # Test 1: Web automation keywords
    print("\n1. Testing web automation keyword discovery:")
    result = await matcher.discover_keywords("click the login button", "web")
    
    if result["success"]:
        print(f"   Found {len(result['matches'])} matches for 'click the login button'")
        for i, match in enumerate(result["matches"][:3]):  # Show top 3
            print(f"   {i+1}. {match['library']}.{match['keyword_name']} (confidence: {match['confidence']:.3f})")
            print(f"      Args: {match['arguments']}")
            print(f"      Doc: {match['documentation'][:80]}...")
    else:
        print(f"   ERROR: {result.get('error', 'Unknown error')}")
    
    # Test 2: Input keywords with better metadata
    print("\n2. Testing input keyword discovery:")
    result = await matcher.discover_keywords("enter text in username field", "web")
    
    if result["success"]:
        print(f"   Found {len(result['matches'])} matches for 'enter text in username field'")
        for i, match in enumerate(result["matches"][:3]):
            print(f"   {i+1}. {match['library']}.{match['keyword_name']} (confidence: {match['confidence']:.3f})")
            print(f"      Args: {match['arguments']}")
            print(f"      Types: {match['argument_types']}")
    else:
        print(f"   ERROR: {result.get('error', 'Unknown error')}")
    
    # Test 3: Verify library loading method
    print("\n3. Testing library loading capabilities:")
    
    # Check how many libraries were loaded
    total_keywords = sum(len(keywords) for keywords in matcher.keyword_registry.values())
    print(f"   Loaded {len(matcher.keyword_registry)} libraries")
    print(f"   Total keywords available: {total_keywords}")
    
    # Show library breakdown
    for lib_name, keywords in matcher.keyword_registry.items():
        print(f"   - {lib_name}: {len(keywords)} keywords")
        
        # Show example keywords with enhanced metadata
        if keywords:
            example = keywords[0]
            print(f"     Example: {example.name}")
            print(f"     Tags: {example.tags}")
            print(f"     Deprecated: {example.deprecated}")
            print(f"     Private: {example.private}")
    
    # Test 4: Tag-based filtering
    print("\n4. Testing tag-based context matching:")
    result = await matcher.discover_keywords("send HTTP request", "api")
    
    if result["success"]:
        print(f"   Found {len(result['matches'])} matches for 'send HTTP request' in API context")
        for i, match in enumerate(result["matches"][:2]):
            print(f"   {i+1}. {match['library']}.{match['keyword_name']} (confidence: {match['confidence']:.3f})")
    else:
        print(f"   ERROR: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("robot.libdoc integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_libdoc_integration())