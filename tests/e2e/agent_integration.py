"""Integration layer between Pydantic AI agents and MCP tools."""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel
from fastmcp import Client

from tests.e2e.metrics_collector import MetricsCollector


@dataclass
class MCPToolContext:
    """Context for MCP tool execution."""

    mcp_client: Client
    metrics_collector: MetricsCollector
    session_id: Optional[str] = None


class MCPAgentIntegration:
    """Integrates Pydantic AI agents with MCP tools."""

    def __init__(self, mcp_client: Client, metrics_collector: MetricsCollector):
        """Initialize the integration.

        Args:
            mcp_client: FastMCP client instance
            metrics_collector: Metrics collector for tracking tool calls
        """
        self.mcp_client = mcp_client
        self.metrics_collector = metrics_collector
        self._tools_cache: Optional[List[Any]] = None

    async def get_available_tools(self) -> List[Any]:
        """Get list of available MCP tools.

        Returns:
            List of MCP tool definitions
        """
        if self._tools_cache is None:
            self._tools_cache = await self.mcp_client.list_tools()
        return self._tools_cache

    async def call_mcp_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call an MCP tool and record metrics.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        args = arguments or {}
        success = False
        result = None
        error = None

        try:
            result = await self.mcp_client.call_tool(tool_name, args)
            success = True
            self.metrics_collector.record_tool_call(
                tool_name=tool_name,
                arguments=args,
                success=success,
                result=result.data if hasattr(result, "data") else result,
            )
        except Exception as e:
            error = str(e)
            self.metrics_collector.record_tool_call(
                tool_name=tool_name,
                arguments=args,
                success=success,
                error=error,
            )
            raise

        return result

    def create_agent_with_mcp_tools(
        self,
        model_name: str = "gpt-4o-mini",
        use_test_model: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Agent:
        """Create a Pydantic AI agent with MCP tools integrated.

        Args:
            model_name: OpenAI model name to use
            use_test_model: Whether to use TestModel instead of real LLM
            system_prompt: Optional custom system prompt

        Returns:
            Configured Pydantic AI Agent
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
Extract the session_id from analyze_scenario and use it in all subsequent calls."""

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            deps_type=MCPToolContext,
        )

        # Register MCP tools as agent tools
        self._register_core_tools(agent)

        return agent

    def _register_core_tools(self, agent: Agent) -> None:
        """Register core MCP tools with the agent.

        Args:
            agent: Pydantic AI agent to register tools with
        """

        @agent.tool
        async def analyze_scenario(
            ctx: RunContext[MCPToolContext], scenario: str, context: str = "web"
        ) -> Dict[str, Any]:
            """Analyze a natural-language scenario and create a session.

            Args:
                ctx: Run context
                scenario: The scenario description
                context: Test context (web, mobile, api, desktop, generic)

            Returns:
                Analysis result with session_id
            """
            result = await self.call_mcp_tool(
                "analyze_scenario", {"scenario": scenario, "context": context}
            )
            # Store session_id in context for reuse
            if hasattr(result, "data") and result.data.get("success"):
                session_id = result.data.get("session_info", {}).get("session_id")
                if session_id:
                    ctx.deps.session_id = session_id
            return result.data if hasattr(result, "data") else result

        @agent.tool
        async def execute_step(
            ctx: RunContext[MCPToolContext],
            keyword: str,
            arguments: List[str],
            session_id: Optional[str] = None,
            assign_to: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Execute a Robot Framework keyword step.

            Args:
                ctx: Run context
                keyword: Keyword name to execute
                arguments: Keyword arguments
                session_id: Session ID (uses context session if not provided)
                assign_to: Optional variable name to assign result to

            Returns:
                Execution result
            """
            sid = session_id or ctx.deps.session_id
            tool_args = {
                "keyword": keyword,
                "arguments": arguments,
            }
            if sid:
                tool_args["session_id"] = sid
            if assign_to:
                tool_args["assign_to"] = assign_to

            result = await self.call_mcp_tool("execute_step", tool_args)
            return result.data if hasattr(result, "data") else result

        @agent.tool
        async def build_test_suite(
            ctx: RunContext[MCPToolContext],
            test_name: str,
            session_id: Optional[str] = None,
            documentation: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Build a Robot Framework test suite from executed steps.

            Args:
                ctx: Run context
                test_name: Name for the test
                session_id: Session ID (uses context session if not provided)
                documentation: Optional test documentation

            Returns:
                Generated test suite
            """
            sid = session_id or ctx.deps.session_id
            tool_args = {"test_name": test_name}
            if sid:
                tool_args["session_id"] = sid
            if documentation:
                tool_args["documentation"] = documentation

            result = await self.call_mcp_tool("build_test_suite", tool_args)
            return result.data if hasattr(result, "data") else result

        @agent.tool
        async def manage_session(
            ctx: RunContext[MCPToolContext],
            action: str,
            session_id: Optional[str] = None,
            **kwargs,
        ) -> Dict[str, Any]:
            """Manage a test session (init, import libraries, set variables).

            Args:
                ctx: Run context
                action: Action to perform (init, import_library, set_variables, etc.)
                session_id: Session ID (uses context session if not provided)
                **kwargs: Additional action-specific parameters

            Returns:
                Action result
            """
            sid = session_id or ctx.deps.session_id
            tool_args = {"action": action}
            if sid:
                tool_args["session_id"] = sid
            tool_args.update(kwargs)

            result = await self.call_mcp_tool("manage_session", tool_args)
            return result.data if hasattr(result, "data") else result

        @agent.tool
        async def recommend_libraries(
            ctx: RunContext[MCPToolContext],
            scenario: str,
            context: str = "web",
            session_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Recommend libraries for a scenario.

            Args:
                ctx: Run context
                scenario: The scenario description
                context: Test context
                session_id: Session ID (uses context session if not provided)

            Returns:
                Library recommendations
            """
            sid = session_id or ctx.deps.session_id
            tool_args = {"scenario": scenario, "context": context}
            if sid:
                tool_args["session_id"] = sid

            result = await self.call_mcp_tool("recommend_libraries", tool_args)
            return result.data if hasattr(result, "data") else result

        @agent.tool
        async def get_session_state(
            ctx: RunContext[MCPToolContext],
            session_id: Optional[str] = None,
            sections: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Get current session state.

            Args:
                ctx: Run context
                session_id: Session ID (uses context session if not provided)
                sections: Optional list of sections to retrieve

            Returns:
                Session state
            """
            sid = session_id or ctx.deps.session_id
            tool_args = {}
            if sid:
                tool_args["session_id"] = sid
            if sections:
                tool_args["sections"] = sections

            result = await self.call_mcp_tool("get_session_state", tool_args)
            return result.data if hasattr(result, "data") else result

    async def run_agent_with_scenario(
        self, agent: Agent, prompt: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Run an agent with a scenario prompt.

        Args:
            agent: Configured Pydantic AI agent
            prompt: Scenario prompt

        Returns:
            Tuple of (agent output, all messages)
        """
        context = MCPToolContext(
            mcp_client=self.mcp_client, metrics_collector=self.metrics_collector
        )

        result = await agent.run(prompt, deps=context)

        # Extract output and messages
        output = result.output if hasattr(result, "output") else str(result)
        messages = result.all_messages() if hasattr(result, "all_messages") else []

        return output, messages
