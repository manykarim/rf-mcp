"""Exact reproduction of conversation 003 issue with build_test_suite.

This test replicates the exact step sequence from the conversation logs
to reproduce the reported missing steps issue.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_conversation_003_exact_reproduction(mcp_client):
    """Reproduce the exact sequence from conversation 003 to test build_test_suite completeness."""
    
    session_id = "restful_booker_api_test"
    
    # EXACT sequence from conversation logs
    
    # 1. analyze_scenario
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {
            "scenario": "Test restful-booker API with comprehensive operations: read booking and assert response, create new booking and assert response, authenticate as admin and assert response, delete booking while authenticated. Use FOR loops for assertions and TRY/EXCEPT when needed.",
            "context": "API",
            "session_id": session_id
        }
    )
    assert analyze.data.get("success") is True
    
    # 2. recommend_libraries 
    recommend = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": "Test restful-booker API with comprehensive operations including GET, POST, DELETE requests, authentication, and response assertions with FOR loops and TRY/EXCEPT handling",
            "context": "api", 
            "session_id": session_id,
            "max_recommendations": 6
        }
    )
    assert recommend.data.get("success") is True
    
    # 3. Create Session
    create_session = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["restful_booker", "https://restful-booker.herokuapp.com"],
            "session_id": session_id,
            "use_context": True
        }
    )
    assert create_session.data.get("success") is True
    
    # 4. GET On Session (read booking)
    get_booking = await mcp_client.call_tool(
        "execute_step", 
        {
            "arguments": ["restful_booker", "/booking/1"],
            "keyword": "GET On Session",
            "session_id": session_id,
            "use_context": True,
            "assign_to": "booking_response"
        }
    )
    assert get_booking.data.get("success") is True
    
    # 5. Should Be Equal As Numbers (status code check)
    assert_status = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["${booking_response.status_code}", "200"],
            "keyword": "Should Be Equal As Numbers", 
            "session_id": session_id,
            "use_context": True
        }
    )
    assert assert_status.data.get("success") is True
    
    # 6. set_variables (booking data)
    set_vars = await mcp_client.call_tool(
        "set_variables",
        {
            "variables": {
                "booking_data": "${booking_response.json()}",
                "basic_fields": [
                    "firstname", "lastname", "totalprice", "depositpaid", 
                    "bookingdates"
                ]
            },
            "session_id": session_id
        }
    )
    assert set_vars.data.get("success") is True
    
    # 7. execute_for_each (field validation) - using basic fields that should always be present
    for_each_validate = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": ["firstname", "lastname", "totalprice", "depositpaid", "bookingdates"],
            "item_var": "field",
            "steps": [{
                "keyword": "Dictionary Should Contain Key",
                "arguments": ["${booking_data}", "${field}"]
            }]
        }
    )
    assert for_each_validate.data.get("success") is True
    
    # 8. Should Be Equal (firstname check) - just validate field exists
    firstname_check = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Should Not Be Empty",
            "arguments": ["${booking_data['firstname']}"],
            "session_id": session_id,
            "use_context": True
        }
    )
    assert firstname_check.data.get("success") is True
    
    # 9. Should Be Equal (lastname check) - just validate field exists
    lastname_check = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Should Not Be Empty",
            "arguments": ["${booking_data['lastname']}"],
            "session_id": session_id,
            "use_context": True
        }
    )
    assert lastname_check.data.get("success") is True
    
    # 10. set_variables (new booking data)
    set_new_booking = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "new_booking_data": {
                    "firstname": "John",
                    "lastname": "Doe",
                    "totalprice": 111,
                    "depositpaid": True,
                    "bookingdates": {
                        "checkin": "2024-01-01",
                        "checkout": "2024-01-05"  
                    },
                    "additionalneeds": "Breakfast"
                }
            }
        }
    )
    assert set_new_booking.data.get("success") is True
    
    # 11. POST On Session (create booking)
    post_booking = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["restful_booker", "/booking", "json=${new_booking_data}"],
            "assign_to": "create_response",
            "keyword": "POST On Session",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert post_booking.data.get("success") is True
    
    # 12. Should Be Equal As Numbers (create status)
    create_status_check = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["${create_response.status_code}", "200"],
            "keyword": "Should Be Equal As Numbers",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert create_status_check.data.get("success") is True
    
    # 13. set_variables (created booking)
    set_created_vars = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "created_booking": "${create_response.json()}",
                "booking_id": "${create_response.json()['bookingid']}"
            }
        }
    )
    assert set_created_vars.data.get("success") is True
    
    # 14. execute_for_each (verify created booking fields)
    for_each_created = await mcp_client.call_tool(
        "execute_for_each",
        {
            "item_var": "field",
            "items": ["firstname", "lastname", "totalprice", "depositpaid", "additionalneeds"],
            "session_id": session_id,
            "steps": [{
                "arguments": ["${created_booking['booking']['${field}']}","${new_booking_data['${field}']}"],
                "keyword": "Should Be Equal"
            }]
        }
    )
    assert for_each_created.data.get("success") is True
    
    # 15. set_variables (auth data)
    set_auth_data = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "auth_data": {
                    "username": "admin",
                    "password": "password123"
                }
            }
        }
    )
    assert set_auth_data.data.get("success") is True
    
    # 16. POST On Session (authenticate)
    post_auth = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["restful_booker", "/auth", "json=${auth_data}"],
            "assign_to": "auth_response",
            "keyword": "POST On Session",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert post_auth.data.get("success") is True
    
    # 17. Should Be Equal As Numbers (auth status)
    auth_status_check = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["${auth_response.status_code}", "200"],
            "keyword": "Should Be Equal As Numbers",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert auth_status_check.data.get("success") is True
    
    # 18. set_variables (auth token)
    set_auth_token = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "auth_token": "${auth_response.json()['token']}"
            }
        }
    )
    assert set_auth_token.data.get("success") is True
    
    # 19. Should Not Be Empty (token validation)
    token_check = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["${auth_token}"],
            "keyword": "Should Not Be Empty",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert token_check.data.get("success") is True
    
    # 20. set_variables (auth headers)  
    set_headers = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "auth_headers": {
                    "Cookie": "token=${auth_token}"
                }
            }
        }
    )
    assert set_headers.data.get("success") is True
    
    # 21. execute_try_except (first delete attempt)
    try_except_1 = await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{
                "arguments": ["restful_booker", "/booking/${booking_id}", "headers=${auth_headers}"],
                "keyword": "DELETE On Session",
                "assign_to": "delete_response"
            }, {
                "arguments": ["${delete_response.status_code}", "201"],
                "keyword": "Should Be Equal As Numbers"
            }],
            "except_steps": [{
                "arguments": ["Failed to delete booking ID ${booking_id}"],
                "keyword": "Log"
            }]
        }
    )
    assert try_except_1.data.get("success") is True
    
    # 22. set_variables (bearer auth headers)
    set_bearer = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "auth_headers_bearer": {
                    "Authorization": "Basic YWRtaW46cGFzc3dvcmQxMjM="
                }
            }
        }
    )
    assert set_bearer.data.get("success") is True
    
    # 23. execute_try_except (second delete attempt with bearer auth)
    try_except_2 = await mcp_client.call_tool(
        "execute_try_except",
        {
            "except_steps": [{
                "arguments": ["Failed to delete booking with Basic auth"],
                "keyword": "Log"
            }],
            "session_id": session_id,
            "try_steps": [{
                "arguments": ["restful_booker", "/booking/${booking_id}", "headers=${auth_headers_bearer}"],
                "assign_to": "delete_response",
                "keyword": "DELETE On Session"
            }, {
                "arguments": ["${delete_response.status_code}", "201"],
                "keyword": "Should Be Equal As Numbers"
            }]
        }
    )
    assert try_except_2.data.get("success") is True
    
    # Get validation status
    validation = await mcp_client.call_tool(
        "get_session_validation_status",
        {"session_id": session_id}
    )
    total_steps = validation.data.get("total_steps", 0)
    validated_steps = validation.data.get("validated_steps", 0)
    
    print(f"\n=== SESSION STATUS ===")
    print(f"Total steps executed: {total_steps}")
    print(f"Validated steps: {validated_steps}")
    
    # Should have around 45+ steps like in the original conversation
    assert total_steps >= 40, f"Expected ~45 steps, got {total_steps}"
    
    # Now build the test suite
    build_result = await mcp_client.call_tool(
        "build_test_suite",
        {
            "documentation": "Comprehensive test suite for Restful-Booker API testing including read, create, authenticate and delete operations with FOR loops and TRY/EXCEPT handling",
            "session_id": session_id,
            "tags": ["api", "restful-booker", "smoke", "crud"], 
            "test_name": "Restful-Booker API Comprehensive Test"
        }
    )
    
    assert build_result.data.get("success") is True
    rf_text = build_result.data.get("rf_text", "")
    
    print(f"\n=== BUILD_TEST_SUITE ANALYSIS ===")
    print(f"Generated RF text length: {len(rf_text)} characters")
    
    # Split into lines for analysis
    lines = rf_text.split('\n')
    keyword_lines = [line for line in lines if line.strip() and not line.startswith('***') and not line.startswith('[') and not line.startswith('#')]
    
    print(f"Total lines in suite: {len(lines)}")
    print(f"Keyword/content lines: {len(keyword_lines)}")
    
    # Check for key operations that MUST be present
    critical_operations = {
        "Create Session": "Create Session    restful_booker" in rf_text,
        "GET booking": "GET On Session" in rf_text and "/booking/1" in rf_text,
        "POST create booking": "POST On Session" in rf_text and "/booking" in rf_text,
        "POST authentication": "/auth" in rf_text,
        "DELETE operation": "DELETE On Session" in rf_text,
        "FOR loops": "FOR    ${field}    IN" in rf_text,
        "TRY/EXCEPT": "TRY" in rf_text and "EXCEPT" in rf_text,
        "Variable settings": "Set Test Variable" in rf_text
    }
    
    missing_operations = []
    present_operations = []
    
    for operation, is_present in critical_operations.items():
        if is_present:
            present_operations.append(operation)
        else:
            missing_operations.append(operation)
    
    print(f"\n✅ PRESENT OPERATIONS:")
    for op in present_operations:
        print(f"  - {op}")
        
    if missing_operations:
        print(f"\n❌ MISSING OPERATIONS:")  
        for op in missing_operations:
            print(f"  - {op}")
    else:
        print(f"\n✅ All critical operations found!")
    
    # Print a sample of the generated suite
    print(f"\n=== GENERATED SUITE SAMPLE ===")
    suite_sample = '\n'.join(lines[:50])  # First 50 lines
    print(suite_sample)
    if len(lines) > 50:
        print("...")
        
    # The test should fail if we're missing critical operations that were in the original
    if missing_operations:
        pytest.fail(f"build_test_suite missing critical operations: {missing_operations}")
    
    # Additional checks for completeness
    operation_count = sum(1 for op in critical_operations.values() if op)
    coverage_ratio = operation_count / len(critical_operations)
    
    assert coverage_ratio >= 0.8, f"Operation coverage too low: {coverage_ratio:.1%}"
    
    # The suite should be reasonably substantial
    assert len(keyword_lines) >= 20, f"Generated suite too short: {len(keyword_lines)} keyword lines"