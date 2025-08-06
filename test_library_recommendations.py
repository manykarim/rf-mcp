#!/usr/bin/env python3
"""Test script for library recommendation feature."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from robotmcp.components.library_recommender import LibraryRecommender

async def test_library_recommendations():
    """Test various scenarios with the library recommender."""
    
    recommender = LibraryRecommender()
    
    test_scenarios = [
        {
            "name": "Web Testing Scenario",
            "scenario": "I want to test a login form on a website, click buttons, and verify page content",
            "context": "web"
        },
        {
            "name": "Mobile App Testing",
            "scenario": "Need to automate an Android app, tap elements, and verify screens",
            "context": "mobile"
        },
        {
            "name": "API Testing Scenario", 
            "scenario": "Test REST API endpoints, send HTTP requests, and validate JSON responses",
            "context": "api"
        },
        {
            "name": "Database Testing",
            "scenario": "Connect to database, execute SQL queries, and verify data",
            "context": "database"
        },
        {
            "name": "Desktop Application",
            "scenario": "Automate a Windows desktop application with dialogs and forms",
            "context": "desktop"
        },
        {
            "name": "Visual Testing",
            "scenario": "Compare images and validate PDF documents",
            "context": "visual"
        },
        {
            "name": "Data-Driven Testing",
            "scenario": "Run tests with data from Excel files and generate fake test data",
            "context": "data"
        },
        {
            "name": "Complex Mixed Scenario",
            "scenario": "Test a web application that connects to a database, generates reports in PDF format, and sends email notifications via API",
            "context": "web"
        }
    ]
    
    print("🤖 Robot Framework Library Recommendation Test\n")
    print("=" * 60)
    
    for i, test_case in enumerate(test_scenarios, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"Scenario: {test_case['scenario']}")
        print(f"Context: {test_case['context']}")
        print("-" * 60)
        
        try:
            result = recommender.recommend_libraries(
                scenario=test_case['scenario'],
                context=test_case['context'],
                max_recommendations=3
            )
            
            if result['success']:
                print(f"✅ Found {len(result['recommendations'])} recommendations")
                print(f"Keywords identified: {', '.join(result['matching_keywords'])}")
                print()
                
                for j, rec in enumerate(result['recommendations'], 1):
                    print(f"  {j}. {rec['library_name']} (Confidence: {rec['confidence']:.2f})")
                    print(f"     Package: {rec['package_name']}")
                    print(f"     Install: {rec['installation_command']}")
                    print(f"     Rationale: {rec['rationale']}")
                    if rec['requires_setup']:
                        print(f"     Setup Required: {', '.join(rec['setup_commands'])}")
                    if rec['platform_requirements']:
                        print(f"     Platform: {', '.join(rec['platform_requirements'])}")
                    print()
                
                # Show installation script
                if result['installation_script'].strip():
                    print("📦 Installation Script:")
                    script_lines = result['installation_script'].split('\n')[:10]  # First 10 lines
                    for line in script_lines:
                        if line.strip():
                            print(f"     {line}")
                    if len(result['installation_script'].split('\n')) > 10:
                        print("     ... (truncated)")
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("\n" + "=" * 60)
    
    # Test edge cases
    print("\n🔍 Edge Case Testing")
    print("=" * 60)
    
    edge_cases = [
        {"scenario": "", "context": "web", "name": "Empty scenario"},
        {"scenario": "xyz abc def", "context": "unknown", "name": "Nonsense input"},
        {"scenario": "test automation", "context": "web", "name": "Generic terms"},
    ]
    
    for case in edge_cases:
        print(f"\nTesting: {case['name']}")
        try:
            result = recommender.recommend_libraries(case['scenario'], case['context'], 2)
            print(f"  Result: {'✅ Success' if result['success'] else '❌ Failed'}")
            print(f"  Recommendations: {len(result.get('recommendations', []))}")
        except Exception as e:
            print(f"  Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_library_recommendations())