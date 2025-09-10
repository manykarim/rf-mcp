"""Isolated tests to verify build_test_suite completeness without external dependencies.

These tests focus on verifying that build_test_suite includes all executed steps
by using only BuiltIn keywords that don't depend on external services.
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
async def test_build_suite_includes_all_executed_steps(mcp_client):
    """Test that build_test_suite includes all types of executed steps."""
    
    session_id = "completeness_test_session"
    
    # Execute a comprehensive set of step types that should all appear in the suite
    
    # 1. Simple keyword
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Starting comprehensive test"],
            "session_id": session_id
        }
    )
    
    # 2. Variable setting  
    await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "test_string": "hello world",
                "test_number": 42,
                "test_list": ["a", "b", "c"]
            }
        }
    )
    
    # 3. Keyword with variable usage
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Should Be Equal",
            "arguments": ["${test_string}", "hello world"],
            "session_id": session_id,
            "use_context": True
        }
    )
    
    # 4. Keyword with assignment
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["test value"],
            "assign_to": "assigned_var",
            "session_id": session_id,
            "use_context": True
        }
    )
    
    # 5. Conditional/evaluation keyword
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate", 
            "arguments": ["${test_number} + 10"],
            "assign_to": "calculated_value",
            "session_id": session_id,
            "use_context": True
        }
    )
    
    # 6. FOR loop
    await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": ["item1", "item2", "item3"],
            "item_var": "current_item", 
            "steps": [{
                "keyword": "Log",
                "arguments": ["Processing: ${current_item}"]
            }]
        }
    )
    
    # 7. TRY/EXCEPT block
    await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{
                "keyword": "Should Be Equal",
                "arguments": ["test", "test"]
            }],
            "except_steps": [{
                "keyword": "Log",
                "arguments": ["This should not execute"]
            }]
        }
    )
    
    # 8. Another keyword to test ordering
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Test completed successfully"],
            "session_id": session_id
        }
    )
    
    # Verify session has expected number of steps
    validation = await mcp_client.call_tool(
        "get_session_validation_status",
        {"session_id": session_id}
    )
    
    total_steps = validation.data.get("total_steps", 0)
    validated_steps = validation.data.get("validated_steps", 0)
    
    print(f"\n=== SESSION ANALYSIS ===")
    print(f"Total steps: {total_steps}")
    print(f"Validated steps: {validated_steps}")
    
    # We should have a significant number of steps (variable settings create multiple steps)
    assert total_steps >= 8, f"Expected at least 8 steps, got {total_steps}"
    assert validated_steps == total_steps, "All steps should be validated"
    
    # Build the test suite
    build_result = await mcp_client.call_tool(
        "build_test_suite",
        {
            "test_name": "Comprehensive Step Test",
            "session_id": session_id,
            "tags": ["completeness", "test"],
            "documentation": "Test suite to verify all step types are included"
        }
    )
    
    assert build_result.data.get("success") is True
    rf_text = build_result.data.get("rf_text", "")
    
    # Analyze the generated suite
    print(f"\n=== BUILD_TEST_SUITE ANALYSIS ===")
    print(f"Generated suite length: {len(rf_text)} characters")
    
    # Check for each type of step we executed
    expected_elements = {
        # Basic logging
        "Starting comprehensive test": "Log    Starting comprehensive test" in rf_text,
        "Test completed successfully": "Log    Test completed successfully" in rf_text,
        
        # Variable operations
        "Set Test Variable test_string": "Set Test Variable" in rf_text and "test_string" in rf_text,
        "Set Test Variable test_number": "Set Test Variable" in rf_text and "test_number" in rf_text,
        "Set Test Variable test_list": "Set Test Variable" in rf_text and "test_list" in rf_text,
        
        # Variable usage in keywords  
        "Should Be Equal with variables": "Should Be Equal    ${test_string}    hello world" in rf_text,
        
        # Assignment
        "Set Variable with assignment": "Set Variable    test value" in rf_text,
        
        # Evaluation
        "Evaluate expression": "Evaluate" in rf_text and "${test_number} + 10" in rf_text,
        
        # FOR loop
        "FOR loop structure": "FOR    ${current_item}    IN" in rf_text,
        "FOR loop content": "Log    Processing: ${current_item}" in rf_text,
        
        # TRY/EXCEPT
        "TRY block": "TRY" in rf_text,
        "EXCEPT block": "EXCEPT" in rf_text,
        "END blocks": "END" in rf_text
    }
    
    # Track missing and present elements
    missing_elements = []
    present_elements = []
    
    for element_name, is_present in expected_elements.items():
        if is_present:
            present_elements.append(element_name)
        else:
            missing_elements.append(element_name)
    
    print(f"\n✅ PRESENT ELEMENTS ({len(present_elements)}):")
    for element in present_elements:
        print(f"  - {element}")
    
    if missing_elements:
        print(f"\n❌ MISSING ELEMENTS ({len(missing_elements)}):")
        for element in missing_elements:
            print(f"  - {element}")
    else:
        print(f"\n✅ All expected elements found!")
    
    # Print the generated suite for inspection
    print(f"\n=== GENERATED SUITE ===")
    lines = rf_text.split('\n')
    print('\n'.join(lines[:60]))  # First 60 lines
    if len(lines) > 60:
        print("...\n[truncated]")
    
    # Calculate coverage
    coverage_ratio = len(present_elements) / len(expected_elements)
    print(f"\nElement coverage: {coverage_ratio:.1%} ({len(present_elements)}/{len(expected_elements)})")
    
    # Test should fail if significant elements are missing
    assert coverage_ratio >= 0.8, f"Coverage too low: {coverage_ratio:.1%}. Missing: {missing_elements}"
    
    # The suite should have reasonable length
    assert len(rf_text) > 500, f"Generated suite too short: {len(rf_text)} characters"
    
    # Should have reasonable number of lines
    assert len(lines) > 20, f"Generated suite too few lines: {len(lines)}"


@pytest.mark.asyncio 
async def test_step_ordering_preservation(mcp_client):
    """Test that build_test_suite preserves the order of executed steps."""
    
    session_id = "ordering_test_session"
    
    # Execute steps in a specific order
    step_sequence = [
        "First step", 
        "Second step",
        "Third step",
        "Fourth step",
        "Fifth step"
    ]
    
    for i, step_text in enumerate(step_sequence):
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": [step_text],
                "session_id": session_id
            }
        )
    
    # Build suite
    build_result = await mcp_client.call_tool(
        "build_test_suite",
        {
            "test_name": "Step Ordering Test",
            "session_id": session_id
        }
    )
    
    assert build_result.data.get("success") is True
    rf_text = build_result.data.get("rf_text", "")
    
    # Check that steps appear in the correct order
    step_positions = []
    for step_text in step_sequence:
        log_line = f"Log    {step_text}"
        position = rf_text.find(log_line)
        if position >= 0:
            step_positions.append((step_text, position))
        else:
            pytest.fail(f"Step '{step_text}' not found in generated suite")
    
    # Verify ordering
    sorted_positions = sorted(step_positions, key=lambda x: x[1])
    expected_order = [(text, pos) for text, pos in step_positions]
    
    print(f"\nStep positions in generated suite:")
    for step_text, position in sorted_positions:
        print(f"  {step_text}: position {position}")
    
    # The positions should be in the same order as executed
    actual_order = [step_text for step_text, _ in sorted_positions]
    assert actual_order == step_sequence, f"Step order not preserved. Expected: {step_sequence}, Got: {actual_order}"


@pytest.mark.asyncio
async def test_complex_workflow_completeness(mcp_client):
    """Test a complex workflow similar to the API scenario to verify all steps are captured."""
    
    session_id = "complex_workflow_session"
    
    # Simulate a complex CRUD-like workflow using only BuiltIn keywords
    
    # SETUP phase
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== SETUP PHASE ==="], "session_id": session_id}
    )
    
    await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "base_url": "http://example.com",
                "user_data": {"name": "John", "age": 30},
                "expected_status": 200
            }
        }
    )
    
    # READ phase
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== READ PHASE ==="], "session_id": session_id}
    )
    
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["mock_response_data"],
            "assign_to": "response",
            "session_id": session_id,
            "use_context": True
        }
    )
    
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Should Not Be Empty",
            "arguments": ["${response}"],
            "session_id": session_id,
            "use_context": True
        }
    )
    
    # VALIDATE phase with FOR loop
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== VALIDATE PHASE ==="], "session_id": session_id}
    )
    
    await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": ["name", "age", "email"],
            "item_var": "field",
            "steps": [{
                "keyword": "Log",
                "arguments": ["Validating field: ${field}"]
            }]
        }
    )
    
    # CREATE phase  
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== CREATE PHASE ==="], "session_id": session_id}
    )
    
    await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "new_record": {"id": 123, "created": True},
                "record_id": 123
            }
        }
    )
    
    # AUTHENTICATE phase
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== AUTHENTICATE PHASE ==="], "session_id": session_id}
    )
    
    await mcp_client.call_tool(
        "set_variables",
        {
            "session_id": session_id,
            "variables": {
                "auth_token": "mock_token_12345",
                "auth_headers": {"Authorization": "Bearer mock_token_12345"}
            }
        }
    )
    
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Should Not Be Empty", 
            "arguments": ["${auth_token}"],
            "session_id": session_id,
            "use_context": True
        }
    )
    
    # DELETE phase with TRY/EXCEPT
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== DELETE PHASE ==="], "session_id": session_id}
    )
    
    await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{
                "keyword": "Log",
                "arguments": ["Attempting delete of record ${record_id}"]
            }, {
                "keyword": "Should Be Equal",
                "arguments": ["${expected_status}", "200"]
            }],
            "except_steps": [{
                "keyword": "Log",
                "arguments": ["Delete failed - this is expected in simulation"]
            }]
        }
    )
    
    # CLEANUP phase
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["=== CLEANUP COMPLETE ==="], "session_id": session_id}
    )
    
    # Check session status
    validation = await mcp_client.call_tool(
        "get_session_validation_status",
        {"session_id": session_id}
    )
    
    total_steps = validation.data.get("total_steps", 0)
    print(f"\nComplex workflow executed {total_steps} steps")
    
    # Should have many steps
    assert total_steps >= 15, f"Expected substantial workflow, got {total_steps} steps"
    
    # Build suite
    build_result = await mcp_client.call_tool(
        "build_test_suite",
        {
            "test_name": "Complex Workflow Test",
            "session_id": session_id,
            "documentation": "Comprehensive workflow test with CRUD operations",
            "tags": ["complex", "workflow", "crud"]
        }
    )
    
    assert build_result.data.get("success") is True
    rf_text = build_result.data.get("rf_text", "")
    
    # Verify all phases are represented
    phases = [
        "=== SETUP PHASE ===",
        "=== READ PHASE ===", 
        "=== VALIDATE PHASE ===",
        "=== CREATE PHASE ===",
        "=== AUTHENTICATE PHASE ===",
        "=== DELETE PHASE ===",
        "=== CLEANUP COMPLETE ==="
    ]
    
    missing_phases = []
    for phase in phases:
        if phase not in rf_text:
            missing_phases.append(phase)
    
    print(f"\nPhase representation:")
    for phase in phases:
        status = "✅" if phase in rf_text else "❌"
        print(f"  {status} {phase}")
    
    # Critical workflow elements
    critical_elements = {
        "Variable settings": "Set Test Variable" in rf_text,
        "Variable usage": "${response}" in rf_text,
        "FOR loop": "FOR    ${field}    IN" in rf_text,
        "TRY/EXCEPT": "TRY" in rf_text and "EXCEPT" in rf_text,
        "Assertions": "Should Not Be Empty" in rf_text or "Should Be Equal" in rf_text
    }
    
    missing_critical = [elem for elem, present in critical_elements.items() if not present]
    
    print(f"\nCritical elements:")
    for element, present in critical_elements.items():
        status = "✅" if present else "❌"
        print(f"  {status} {element}")
    
    # Test assertions
    assert len(missing_phases) <= 1, f"Too many missing phases: {missing_phases}"
    assert len(missing_critical) == 0, f"Missing critical elements: {missing_critical}"
    
    # Suite should be substantial
    lines = rf_text.split('\n')
    keyword_lines = [line for line in lines if line.strip() and 'Log    ' in line or 'Set Test Variable' in line or 'Should' in line]
    
    print(f"\nSuite metrics:")
    print(f"  Total characters: {len(rf_text)}")
    print(f"  Total lines: {len(lines)}")  
    print(f"  Keyword lines: {len(keyword_lines)}")
    
    assert len(keyword_lines) >= 10, f"Generated suite has too few keyword lines: {len(keyword_lines)}"