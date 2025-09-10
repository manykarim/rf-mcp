"""Unit tests to reproduce and verify the build_test_suite missing steps bug.

This test reproduces the exact scenario from the conversation where build_test_suite
was missing critical CRUD operations despite all steps being successfully executed.
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
async def test_restful_booker_build_suite_missing_steps_bug(mcp_client):
    """Reproduce the exact scenario from conversation 003 where build_test_suite missed steps.
    
    This test replicates the restful-booker API testing scenario that resulted in 
    a build_test_suite response missing critical CRUD operations.
    """
    session_id = "restful_booker_api_test_bug_reproduction"
    
    # Step 1: Analyze scenario
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {
            "scenario": "Test restful-booker API with comprehensive operations: read booking and assert response, create new booking and assert response, authenticate as admin and assert response, delete booking while authenticated. Use FOR loops for assertions and TRY/EXCEPT when needed.",
            "context": "API",
            "session_id": session_id
        }
    )
    assert analyze.data.get("success") is True
    
    # Step 2: Recommend libraries
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
    
    # Step 3: Execute the key CRUD operations that should be included in build_test_suite
    
    # Create Session
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
    
    # READ: Get a booking
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
    
    # Assert status code
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
    
    # Set variables for booking data  
    set_booking_vars = await mcp_client.call_tool(
        "set_variables",
        {
            "variables": {
                "booking_data": "${booking_response.json()}",
                "basic_fields": ["firstname", "lastname", "totalprice", "depositpaid", "bookingdates"]
            },
            "session_id": session_id
        }
    )
    assert set_booking_vars.data.get("success") is True
    
    # FOR loop to validate basic fields that should always be present
    for_each_validation = await mcp_client.call_tool(
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
    assert for_each_validation.data.get("success") is True
    assert for_each_validation.data.get("count") == 5
    
    # CREATE: Set new booking data
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
    
    # CREATE: Post new booking
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
    
    # Assert create status
    assert_create_status = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["${create_response.status_code}", "200"],
            "keyword": "Should Be Equal As Numbers",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert assert_create_status.data.get("success") is True
    
    # Set created booking variables
    set_created_vars = await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "created_booking": "${create_response.json()}",
                "booking_id": "${create_response.json()['bookingid']}",
                "created_booking_fields": ["firstname", "lastname", "totalprice", "depositpaid", "bookingdates", "additionalneeds"]
            }
        }
    )
    assert set_created_vars.data.get("success") is True
    
    # Validate the created booking has our expected fields (since we control the creation)
    validate_created_booking = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": ["firstname", "lastname", "totalprice", "depositpaid", "bookingdates", "additionalneeds"],
            "item_var": "created_field",
            "steps": [{
                "keyword": "Dictionary Should Contain Key",
                "arguments": ["${created_booking['booking']}", "${created_field}"]
            }]
        }
    )
    assert validate_created_booking.data.get("success") is True
    assert validate_created_booking.data.get("count") == 6
    
    # AUTHENTICATE: Set auth data
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
    
    # AUTHENTICATE: Post authentication
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
    
    # Assert auth status
    assert_auth_status = await mcp_client.call_tool(
        "execute_step",
        {
            "arguments": ["${auth_response.status_code}", "200"],
            "keyword": "Should Be Equal As Numbers",
            "session_id": session_id,
            "use_context": True
        }
    )
    assert assert_auth_status.data.get("success") is True
    
    # Set auth token
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
    
    # DELETE: Set auth headers and try delete with TRY/EXCEPT
    set_auth_headers = await mcp_client.call_tool(
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
    assert set_auth_headers.data.get("success") is True
    
    # TRY/EXCEPT delete
    try_except_delete = await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{
                "arguments": ["restful_booker", "/booking/${booking_id}", "headers=${auth_headers_bearer}"],
                "assign_to": "delete_response",
                "keyword": "DELETE On Session"
            }, {
                "arguments": ["${delete_response.status_code}", "201"],
                "keyword": "Should Be Equal As Numbers"
            }],
            "except_steps": [{
                "arguments": ["Failed to delete booking with Basic auth"],
                "keyword": "Log"
            }]
        }
    )
    assert try_except_delete.data.get("success") is True
    
    # Step 4: Check session validation status
    validation_status = await mcp_client.call_tool(
        "get_session_validation_status",
        {"session_id": session_id}
    )
    assert validation_status.data.get("success") is True
    total_steps = validation_status.data.get("total_steps", 0)
    validated_steps = validation_status.data.get("validated_steps", 0)
    print(f"Session has {total_steps} total steps, {validated_steps} validated")
    
    # Should have many steps (similar to the 45 in the conversation)
    assert total_steps > 15, f"Expected many steps, got {total_steps}"
    assert validated_steps == total_steps, f"All steps should be validated"
    
    # Step 5: Build test suite - THIS IS WHERE THE BUG OCCURS
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
    print(f"Generated RF text length: {len(rf_text)}")
    
    # Step 6: VERIFY THE BUG - Check what's missing from the generated suite
    
    # These should be present but may be missing due to the bug:
    missing_elements = []
    
    # Check for CREATE operation (POST new booking)
    if "POST On Session" not in rf_text or "/booking" not in rf_text:
        missing_elements.append("POST new booking operation")
    
    # Check for AUTHENTICATE operation (POST auth)
    if "/auth" not in rf_text:
        missing_elements.append("POST authentication operation")
        
    # Check for variable settings
    if "new_booking_data" not in rf_text:
        missing_elements.append("new_booking_data variable")
        
    if "auth_data" not in rf_text:
        missing_elements.append("auth_data variable")
    
    # Check for Create Session
    if "Create Session" not in rf_text:
        missing_elements.append("Create Session operation")
        
    # Check for GET booking operation
    if "GET On Session" not in rf_text:
        missing_elements.append("GET booking operation")
    
    # Print detailed analysis
    print("\n=== BUILD_TEST_SUITE ANALYSIS ===")
    print(f"Total executed steps: {total_steps}")
    print(f"Generated RF suite length: {len(rf_text)}")
    
    if missing_elements:
        print(f"\n❌ MISSING ELEMENTS DETECTED:")
        for element in missing_elements:
            print(f"  - {element}")
    else:
        print(f"\n✅ All expected elements found in generated suite")
        
    print(f"\n=== GENERATED RF SUITE ===")
    print(rf_text[:1000] + "..." if len(rf_text) > 1000 else rf_text)
    
    # The test fails if critical elements are missing, confirming the bug
    if missing_elements:
        pytest.fail(f"build_test_suite is missing critical elements: {missing_elements}")


@pytest.mark.asyncio 
async def test_simple_workflow_build_suite_completeness(mcp_client):
    """Test a simple workflow to ensure build_test_suite includes all steps."""
    session_id = "simple_completeness_test"
    
    # Execute a few simple steps
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log", 
            "arguments": ["Starting test"],
            "session_id": session_id
        }
    )
    
    await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {"test_var": "test_value"}
        }
    )
    
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Should Be Equal",
            "arguments": ["${test_var}", "test_value"], 
            "session_id": session_id,
            "use_context": True
        }
    )
    
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Test completed"],
            "session_id": session_id
        }
    )
    
    # Check session has 4+ steps
    validation = await mcp_client.call_tool(
        "get_session_validation_status", 
        {"session_id": session_id}
    )
    total_steps = validation.data.get("total_steps", 0)
    assert total_steps >= 4
    
    # Build suite
    build_result = await mcp_client.call_tool(
        "build_test_suite",
        {
            "test_name": "Simple Completeness Test",
            "session_id": session_id
        }
    )
    
    assert build_result.data.get("success") is True
    rf_text = build_result.data.get("rf_text", "")
    
    # All executed operations should be in the suite
    assert "Log    Starting test" in rf_text
    assert "Should Be Equal    ${test_var}    test_value" in rf_text  
    assert "Log    Test completed" in rf_text
    
    # Variable setting should result in Set Test Variable
    assert "Set Test Variable" in rf_text or "test_var" in rf_text


@pytest.mark.asyncio
async def test_build_suite_step_count_validation(mcp_client):
    """Validate that build_test_suite includes a reasonable number of steps relative to executed steps."""
    session_id = "step_count_validation"
    
    # Execute 10 distinct steps
    steps_executed = 0
    
    for i in range(3):
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": [f"Step {i+1}"],
                "session_id": session_id
            }
        )
        steps_executed += 1
        
    await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id, 
            "variables": {f"var_{i}": f"value_{i}" for i in range(3)}
        }
    )
    steps_executed += 3  # Variable setting creates steps
    
    for i in range(2):
        await mcp_client.call_tool(
            "execute_step", 
            {
                "keyword": "Should Be Equal",
                "arguments": [f"${{var_{i}}}", f"value_{i}"],
                "session_id": session_id,
                "use_context": True
            }
        )
        steps_executed += 1
    
    # Get actual step count
    validation = await mcp_client.call_tool(
        "get_session_validation_status",
        {"session_id": session_id}
    )
    actual_steps = validation.data.get("total_steps", 0)
    
    # Build suite
    build_result = await mcp_client.call_tool(
        "build_test_suite",
        {
            "test_name": "Step Count Validation",
            "session_id": session_id
        }
    )
    
    rf_text = build_result.data.get("rf_text", "")
    
    # Count keyword lines in generated suite (rough approximation)
    rf_lines = rf_text.split('\n')
    keyword_lines = [line for line in rf_lines if line.strip() and not line.startswith('***') and not line.startswith('[') and not line.startswith('#')]
    rf_keyword_count = len([line for line in keyword_lines if any(kw in line for kw in ['Log', 'Should Be Equal', 'Set Test Variable', 'Set Variable'])])
    
    print(f"Executed steps: {actual_steps}, RF keyword lines: {rf_keyword_count}")
    
    # The ratio should be reasonable - RF suite should have a substantial portion of the executed steps
    ratio = rf_keyword_count / actual_steps if actual_steps > 0 else 0
    
    # Allow some variance but expect at least 50% of steps to appear
    assert ratio >= 0.5, f"Generated suite has too few steps: {rf_keyword_count}/{actual_steps} = {ratio:.2%}"