"""E2E test for DemoShop scenario with autonomous agent."""

import os
import pytest
import pytest_asyncio

from tests.e2e.fixtures import mcp_server, metrics_collector
from tests.e2e.agent_integration import MCPAgentIntegration


def should_use_real_llm() -> bool:
    """Check if tests should use real LLM."""
    use_real = os.getenv("USE_REAL_LLM", "false").lower()
    return use_real in ("true", "1", "yes")


def get_model_name() -> str:
    """Get the model name to use for testing.

    Reads OPENAI_MODEL environment variable at call time (not import time)
    to ensure environment changes are picked up.
    """
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return model


@pytest.mark.asyncio
@pytest.mark.skipif(
    not should_use_real_llm(),
    reason="Requires USE_REAL_LLM=true and OPENAI_API_KEY to be set",
)
async def test_demoshop_checkout_scenario(mcp_server, metrics_collector):
    """Test DemoShop checkout scenario with autonomous agent."""
    
    integration = MCPAgentIntegration(mcp_server, metrics_collector)
    
    # Create agent with real LLM
    model_name = get_model_name()
    agent = integration.create_agent_with_mcp_tools(
        model_name=model_name, use_test_model=False
    )
    
    # Start metrics collection
    metrics_collector.start_recording()
    
    # Define the scenario prompt (NOTE: No library specified - agent should choose)
    prompt = """Use RobotMCP to create a TestSuite and execute it step wise.

- Open https://demoshop.makrocode.de/ with firefox and headless=False
- Add item to cart
- Assert item was added to cart
- Add another item to cart
- Assert another item was added to cart
- Complete Checkout
- Assert checkout was successful

Execute step by step and build final test suite afterwards"""

    print(f"\n=== DemoShop Checkout Scenario ===")
    print(f"Model: {model_name}")
    print(f"Scenario: {prompt[:100]}...")
    
    # Run agent
    try:
        output, messages = await integration.run_agent_with_scenario(agent, prompt)
    except Exception as e:
        metrics_collector.stop_recording()
        pytest.fail(f"Agent execution failed: {e}")
    
    # Stop metrics collection
    metrics_collector.stop_recording()
    
    # Print results
    tool_calls = metrics_collector.tool_calls
    tool_names = [tc.tool_name for tc in tool_calls]
    
    print(f"\nTool calls made: {tool_names}")
    print(f"Total tool calls: {len(tool_calls)}")
    print(f"Agent output: {output[:200]}...")
    
    # Check for errors
    errors = [tc for tc in tool_calls if not tc.success]
    
    if errors:
        print("\n=== ERRORS DETECTED ===")
        for i, error_call in enumerate(errors, 1):
            print(f"{i}. Tool: {error_call.tool_name}")
            print(f"   Arguments: {error_call.arguments}")
            if error_call.error:
                print(f"   Error: {error_call.error[:200]}")
    
    # Basic assertions
    assert len(tool_calls) > 0, "Agent should make at least one tool call"
    assert "analyze_scenario" in tool_names, "Agent should call analyze_scenario"
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Total tool calls: {len(tool_calls)}")
    print(f"Successful: {len([tc for tc in tool_calls if tc.success])}")
    print(f"Failed: {len(errors)}")
    
    # For this test, we just want to see the behavior, not enforce success
    if errors:
        print(f"\n⚠️  Test completed with {len(errors)} errors (expected for complex scenario)")
    else:
        print(f"\n✅ Test completed successfully")
