"""Integration layer between Pydantic AI agents and MCP tools using FastMCPToolset."""

import asyncio
from typing import Any, Dict, Optional
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.usage import UsageLimits
from fastmcp import FastMCP

from tests.e2e.metrics_collector import MetricsCollector


@dataclass
class MCPToolContext:
    """Context for MCP tool execution."""
    metrics_collector: MetricsCollector
    session_id: Optional[str] = None


class MCPAgentIntegration:
    """Integrates Pydantic AI agents with MCP tools using FastMCPToolset.

    This class uses Pydantic AI's native MCP support via FastMCPToolset,
    which automatically discovers and exposes ALL MCP server tools to the agent.

    Benefits over manual wrappers:
    - Always in sync with MCP server (no manual updates needed)
    - Uses actual MCP tool docstrings (accurate LLM guidance)
    - Includes ALL MCP tools automatically (complete coverage)
    - Zero maintenance burden (MCP changes propagate automatically)
    """

    def __init__(self, mcp_server: FastMCP, metrics_collector: MetricsCollector):
        """Initialize the integration.

        Args:
            mcp_server: FastMCP server instance to expose as tools
            metrics_collector: Metrics collector for tracking tool calls
        """
        self.mcp_server = mcp_server
        self.metrics_collector = metrics_collector
        self._toolset: Optional[FastMCPToolset] = None

    async def get_toolset(self) -> FastMCPToolset:
        """Get or create the FastMCP toolset.

        Returns:
            FastMCPToolset instance wrapping the MCP server
        """
        if self._toolset is None:
            # Create toolset from FastMCP server - this automatically discovers
            # all tools defined in the MCP server
            self._toolset = FastMCPToolset(self.mcp_server)
        return self._toolset

    def create_agent_with_mcp_tools(
        self,
        model_name: str = "gpt-5-mini",
        use_test_model: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Agent:
        """Create a Pydantic AI agent with MCP tools integrated via FastMCPToolset.

        Args:
            model_name: OpenAI model name to use
            use_test_model: Whether to use TestModel instead of real LLM
            system_prompt: Optional custom system prompt

        Returns:
            Configured Pydantic AI Agent with ALL MCP tools available
        """
        if use_test_model:
            model = TestModel()
        else:
            model = OpenAIModel(model_name)

        if system_prompt is None:
            system_prompt = """You are a test automation assistant using Robot Framework MCP server.
You have access to MCP tools for creating and executing Robot Framework tests.

When given a test scenario:
1. Use analyze_scenario to understand the requirements and create a session
2. Use execute_step to build the test step by step
3. Use build_test_suite to generate the final test suite

Always use the appropriate MCP tools to accomplish testing tasks.
Extract the session_id from analyze_scenario and use it in all subsequent calls.

IMPORTANT: Common Robot Framework keywords are in these libraries:
- BuiltIn: Create List, Log, Should Be Equal, Should Contain, Set Variable, Length Should Be, etc.
- Collections: Sort List, Append To List, Get From List, List Should Contain Value, etc.
- String: Convert To String, Split String, Replace String, etc.
- XML: Parse XML, Get Element, Get Element Text, Get Element Attribute, Elements Should Match, etc.
- OperatingSystem: Create File, File Should Exist, Get File, etc.

When using keywords, call them WITHOUT library prefix (e.g., 'Create List' not 'BuiltIn.Create List' or 'Collections.Create List').
The 'Create List' keyword is in BuiltIn, NOT in Collections."""

        # Create toolset with metrics wrapping
        toolset = self._create_metrics_wrapped_toolset()

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            deps_type=MCPToolContext,
            toolsets=[toolset],
        )

        return agent

    def _create_metrics_wrapped_toolset(self, max_retries: int = 5) -> FastMCPToolset:
        """Create a FastMCPToolset with metrics collection wrapper.

        This wraps the FastMCPToolset to collect metrics on tool calls
        while preserving all MCP server tools and their signatures.

        Args:
            max_retries: Maximum retry attempts for failed tool calls.
                When a tool fails, the error message is sent back to the model
                which can then correct its approach and retry. Default is 3
                to allow recovery from common mistakes like wrong syntax.

        Returns:
            Metrics-wrapped FastMCPToolset
        """
        # Create the base toolset from our MCP server with retry support
        # max_retries allows the agent to recover from errors by seeing
        # the error message and correcting its approach
        base_toolset = FastMCPToolset(self.mcp_server, max_retries=max_retries)

        # Wrap it to add metrics collection
        # Note: FastMCPToolset handles all tool discovery and execution automatically
        # We just need to add metrics tracking on top
        return self._wrap_toolset_with_metrics(base_toolset)

    def _wrap_toolset_with_metrics(self, toolset: FastMCPToolset) -> FastMCPToolset:
        """Wrap a toolset to collect metrics on tool calls.

        Args:
            toolset: Base toolset to wrap

        Returns:
            Metrics-wrapped toolset
        """
        # Store original call_tool method
        original_call_tool = toolset.call_tool
        metrics_collector = self.metrics_collector

        # Wrap call_tool to add metrics
        async def metrics_wrapped_call_tool(name: str, tool_args: Dict[str, Any], ctx: Any, tool: Any) -> Any:
            """Wrapped call_tool that records metrics."""
            success = False
            result = None
            error = None

            try:
                result = await original_call_tool(name, tool_args, ctx, tool)
                success = True
                metrics_collector.record_tool_call(
                    tool_name=name,
                    arguments=tool_args,
                    success=success,
                    result=result if isinstance(result, dict) else {"output": str(result)},
                )
            except Exception as e:
                error = str(e)
                metrics_collector.record_tool_call(
                    tool_name=name,
                    arguments=tool_args,
                    success=success,
                    error=error,
                )
                raise

            return result

        # Replace the call_tool method
        toolset.call_tool = metrics_wrapped_call_tool

        return toolset

    async def run_agent_with_scenario(self, agent: Agent, prompt: str):
        """Run agent with a scenario prompt.

        Args:
            agent: Pydantic AI agent to run
            prompt: Scenario prompt to give to the agent

        Returns:
            Tuple of (output string, messages list)
        """
        # Create context for the agent run
        context = MCPToolContext(
            metrics_collector=self.metrics_collector,
            session_id=None
        )

        # Run the agent with increased request limit for complex scenarios
        # Default pydantic-ai limit is 50, but complex scenarios with error recovery
        # may need more iterations
        usage_limits = UsageLimits(request_limit=100)
        result = await agent.run(prompt, deps=context, usage_limits=usage_limits)

        # Extract output and messages
        output = result.data if hasattr(result, 'data') else str(result)
        messages = result.all_messages() if hasattr(result, 'all_messages') else []

        return output, messages


async def create_agent_from_mcp_server(
    mcp_server: FastMCP,
    metrics_collector: MetricsCollector,
    model_name: str = "gpt-5-mini",
    use_test_model: bool = False,
    system_prompt: Optional[str] = None,
) -> Agent:
    """Helper function to quickly create an agent from an MCP server.

    Args:
        mcp_server: FastMCP server instance
        metrics_collector: Metrics collector
        model_name: OpenAI model name
        use_test_model: Whether to use test model
        system_prompt: Optional custom system prompt

    Returns:
        Configured Pydantic AI agent with all MCP tools
    """
    integration = MCPAgentIntegration(mcp_server, metrics_collector)
    return integration.create_agent_with_mcp_tools(
        model_name=model_name,
        use_test_model=use_test_model,
        system_prompt=system_prompt,
    )
