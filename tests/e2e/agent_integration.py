"""Integration layer between Pydantic AI agents and MCP tools using FastMCPToolset."""

import asyncio
from typing import Any, Dict, Optional
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.usage import UsageLimits
from fastmcp import FastMCP

from tests.e2e.metrics_collector import MetricsCollector
from tests.e2e.minimax_support import resolve_model


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

    # A DELIBERATELY NEUTRAL system prompt. It must NOT restate rf-mcp's own tool
    # guidance (which tools to call, in what order, session_id threading, keyword
    # libraries) — doing so masks the quality of rf-mcp's tool descriptions and MCP
    # instructions, defeating the whole point of the agentic e2e gate. The agent must
    # rely on rf-mcp's ACTUAL tool descriptions + injected server instructions, so that
    # degrading them is observable in the metrics.
    NEUTRAL_SYSTEM_PROMPT = (
        "You are a test-automation agent. You have access to a set of MCP tools for "
        "building and running Robot Framework tests. Read each tool's description and "
        "the server guidance below, then discover and use the appropriate tools to "
        "accomplish the user's task. Follow the tools' own instructions for sequencing, "
        "sessions, arguments and keyword usage — do not rely on outside knowledge of the "
        "tool set."
    )

    def _resolve_system_prompt(self, system_prompt: Optional[str]) -> str:
        """Build the agent system prompt: neutral base + the server's MCP instructions.

        The rf-mcp server instructions (the WORKFLOW GUIDE) are injected here because
        FastMCPToolset forwards only per-tool descriptions, not the server's
        ``instructions`` — so without this the agent never sees the MCP instructions and
        the gate cannot measure their quality. Reading them from the live server object
        means degrading them (e.g. ROBOTMCP_INSTRUCTIONS=off) is reflected in the agent's
        behaviour and the metrics.
        """
        if system_prompt is not None:
            return system_prompt
        server_instructions = (getattr(self.mcp_server, "instructions", "") or "").strip()
        if server_instructions:
            return (
                f"{self.NEUTRAL_SYSTEM_PROMPT}\n\n"
                f"--- MCP server guidance ---\n{server_instructions}"
            )
        return self.NEUTRAL_SYSTEM_PROMPT

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
            system_prompt: Optional custom system prompt. When None, a NEUTRAL prompt is
                used plus the server's own MCP instructions — so the agent depends on
                rf-mcp's real tool descriptions/instructions, which is what the gate
                measures.

        Returns:
            Configured Pydantic AI Agent with ALL MCP tools available
        """
        if use_test_model:
            model = TestModel()
        else:
            # resolve_model routes MiniMax model IDs to the MiniMax OpenAI-compatible
            # endpoint (MINIMAX_API_KEY); gpt-* keep the default OpenAI provider.
            model = resolve_model(model_name)

        system_prompt = self._resolve_system_prompt(system_prompt)

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

    async def run_agent_with_scenario(
        self, agent: Agent, prompt: str, request_limit: int = 100
    ):
        """Run agent with a scenario prompt.

        Args:
            agent: Pydantic AI agent to run
            prompt: Scenario prompt to give to the agent
            request_limit: Max model requests before UsageLimitExceeded. Lower it for
                weaker/looping models (e.g. MiniMax M2) to bound cost and wall-clock.

        Returns:
            Tuple of (output string, messages list)
        """
        # Create context for the agent run
        context = MCPToolContext(
            metrics_collector=self.metrics_collector,
            session_id=None
        )

        # Run the agent with a bounded request limit for complex scenarios.
        # Default pydantic-ai limit is 50, but complex scenarios with error recovery
        # may need more iterations.
        usage_limits = UsageLimits(request_limit=request_limit)
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
