"""Core orchestration logic for running MCP prompts via an OpenAI model."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from fastmcp import Client
from fastmcp.client import FastMCPTransport
from mcp import types as mcp_types

from robotmcp.prompt.config import PromptRuntimeConfig
from robotmcp.prompt.llm_client import LlmResponse, OpenAILlmClient
from robotmcp.prompt.tool_schema import get_openai_tools
from robotmcp.server import mcp

logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class PromptExecutionError(RuntimeError):
    """Raised when prompt execution fails fatally."""


class LlmClientProtocol(Protocol):
    def complete_chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> LlmResponse:  # pragma: no cover - Protocol definition
        ...


@dataclass
class ExecutedToolCall:
    name: str
    arguments: Dict[str, Any]
    success: bool
    response_text: str


@dataclass
class PromptRunResult:
    success: bool
    final_response: str
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    executed_calls: List[ExecutedToolCall] = field(default_factory=list)
    iterations: int = 0


class PromptRunner:
    """Coordinates MCP client, OpenAI model, and Robot session defaults."""

    def __init__(
        self,
        *,
        llm_client_factory: Callable[[PromptRuntimeConfig], LlmClientProtocol] | None = None,
        client_builder: Callable[[], Client] | None = None,
    ) -> None:
        self._llm_client_factory = llm_client_factory or self._default_llm_factory
        self._client_builder = client_builder or self._default_client_builder

    def run(
        self,
        *,
        scenario: str,
        prompt_key: str,
        session_id: str,
        config: PromptRuntimeConfig,
        prelude_messages: List[Dict[str, Any]] | None = None,
    ) -> PromptRunResult:
        """Synchronously run the prompt using asyncio."""

        return asyncio.run(
            self._run_prompt_async(
                scenario=scenario,
                prompt_key=prompt_key,
                session_id=session_id,
                config=config,
                prelude_messages=prelude_messages,
            )
        )

    def run_chat(
        self,
        *,
        message: str,
        session_id: str,
        config: PromptRuntimeConfig,
        prelude_messages: List[Dict[str, Any]] | None = None,
    ) -> PromptRunResult:
        """Run a free-form chat interaction using the MCP toolset."""

        return asyncio.run(
            self._run_chat_async(
                message=message,
                session_id=session_id,
                config=config,
                prelude_messages=prelude_messages,
            )
        )

    async def _run_prompt_async(
        self,
        *,
        scenario: str,
        prompt_key: str,
        session_id: str,
        config: PromptRuntimeConfig,
        prelude_messages: List[Dict[str, Any]] | None = None,
    ) -> PromptRunResult:
        async with self._client_builder() as client:
            base_messages = await self._render_prompt(client, prompt_key, scenario)
            return await self._run_conversation(
                client=client,
                initial_messages=base_messages,
                scenario=scenario,
                session_id=session_id,
                config=config,
                prelude_messages=prelude_messages,
            )

    async def _run_chat_async(
        self,
        *,
        message: str,
        session_id: str,
        config: PromptRuntimeConfig,
        prelude_messages: List[Dict[str, Any]] | None = None,
    ) -> PromptRunResult:
        async with self._client_builder() as client:
            base_messages = [
                {
                    "role": "user",
                    "content": message,
                }
            ]
            return await self._run_conversation(
                client=client,
                initial_messages=base_messages,
                scenario=message,
                session_id=session_id,
                config=config,
                prelude_messages=prelude_messages,
            )

    async def _run_conversation(
        self,
        *,
        client: Client,
        initial_messages: List[Dict[str, Any]],
        scenario: str,
        session_id: str,
        config: PromptRuntimeConfig,
        prelude_messages: List[Dict[str, Any]] | None = None,
    ) -> PromptRunResult:
        llm_client = self._llm_client_factory(config)
        executed_calls: List[ExecutedToolCall] = []
        transcript: List[Dict[str, Any]] = []
        iteration = 0

        tools = await client.list_tools()
        openai_tools = get_openai_tools(tools)
        if not openai_tools:
            raise PromptExecutionError("No MCP tools available for prompt execution")

        messages: List[Dict[str, Any]] = [
            self._system_instruction_message(),
            self._workflow_instruction_message(),
        ]
        if prelude_messages:
            messages.extend(prelude_messages)
        messages.extend(initial_messages)
        transcript.extend(messages)

        scenario_hint_used = False

        while iteration < config.max_iterations:
            iteration += 1
            llm_response = llm_client.complete_chat(
                model=config.model,
                messages=messages,
                tools=openai_tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            assistant_record: Dict[str, Any] = {
                "role": "assistant",
                "content": llm_response.content,
            }
            if llm_response.tool_calls:
                assistant_record["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments or {}),
                        },
                    }
                    for call in llm_response.tool_calls
                ]
            messages.append(assistant_record)
            transcript.append(assistant_record)

            if not llm_response.tool_calls:
                final = llm_response.content.strip()
                return PromptRunResult(
                    success=bool(final),
                    final_response=final or "LLM returned an empty response",
                    transcript=transcript,
                    executed_calls=executed_calls,
                    iterations=iteration,
                )

            for call in llm_response.tool_calls:
                args = dict(call.arguments or {})
                self._inject_defaults(call.name, args, session_id, scenario, scenario_hint_used)
                if call.name == "execute_step" and not scenario_hint_used:
                    scenario_hint_used = True

                logger.info("MCP prompt executing tool %s with args %s", call.name, args)
                tool_result = await client.call_tool_mcp(call.name, args)
                response_text = _render_tool_result(tool_result)
                success = not getattr(tool_result, "isError", False)
                executed_calls.append(
                    ExecutedToolCall(
                        name=call.name,
                        arguments=args,
                        success=success,
                        response_text=response_text,
                    )
                )

                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": response_text,
                }
                messages.append(tool_message)
                transcript.append(tool_message)

                # Allow the LLM to inspect the failure and retry in later turns

        return PromptRunResult(
            success=False,
            final_response="Maximum prompt iterations exceeded",
            transcript=transcript,
            executed_calls=executed_calls,
            iterations=iteration,
        )

    async def _render_prompt(
        self, client: Client, prompt_key: str, scenario: str
    ) -> List[Dict[str, Any]]:
        prompt_result = await client.get_prompt(prompt_key, {"scenario": scenario})
        messages: List[Dict[str, Any]] = []
        for msg in getattr(prompt_result, "messages", []) or []:
            content = msg.content
            if isinstance(content, list):
                text = "\n".join(
                    block.text
                    for block in content
                    if isinstance(block, mcp_types.TextContent)
                )
            elif hasattr(content, "text"):
                text = content.text  # type: ignore[attr-defined]
            else:
                text = str(content or "")
            messages.append({"role": msg.role, "content": text})
        if not messages:
            messages.append(
                {
                    "role": "system",
                    "content": f"Use RobotMCP tools to automate this scenario: {scenario}",
                }
            )
        return messages

    def _inject_defaults(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session_id: str,
        scenario: str,
        scenario_hint_used: bool,
    ) -> None:
        if tool_name in {
            "manage_session",
            "execute_step",
            "get_session_state",
            "build_test_suite",
            "run_test_suite",
            "find_keywords",
            "get_keyword_info",
        }:
            args.setdefault("session_id", session_id)
        if tool_name == "execute_step":
            args.setdefault("use_context", True)
            args.setdefault("detail_level", "minimal")
            # Allow the Agent to handle failures (e.g. retries) instead of stopping execution
            args.setdefault("raise_on_failure", False)
            if not scenario_hint_used:
                args.setdefault("scenario_hint", scenario)
        if tool_name == "manage_session":
            args.setdefault("action", "init")
        if tool_name == "analyze_scenario":
            args.setdefault("context", "web")
        if tool_name == "recommend_libraries":
            args.setdefault("context", "web")
        if tool_name == "get_session_state":
            args.setdefault("sections", ["rf_context", "variables"])
        if tool_name == "build_test_suite":
            args.setdefault("test_name", "Prompt Generated Test")

    def _system_instruction_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are operating within RobotMCP. Always supply the provided session_id "
                "when calling tools so steps execute inside the active Robot Framework session."
            ),
        }

    def _workflow_instruction_message(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "Follow this workflow: 1) call analyze_scenario to create or reuse a session, "
                "2) call recommend_libraries and check_library_availability, 3) initialize sessions via manage_session, "
                "4) drive execution using execute_step (use_context=True). "
                "CRITICAL: If a step fails or you are unsure of the state, call get_session_state(sections=['page_source', 'application_state']) "
                "to inspect the DOM or application state before retrying. "
                "5) Once the scenario is complete, ALWAYS call build_test_suite to generate the final Robot Framework code "
                "and include it in your final response."
            ),
        }

    def _default_llm_factory(self, config: PromptRuntimeConfig) -> LlmClientProtocol:
        return OpenAILlmClient(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def _default_client_builder(self) -> Client:
        return Client(FastMCPTransport(mcp, raise_exceptions=True))


def _render_tool_result(result: Any) -> str:
    """Convert CallToolResult content to a loggable, serializable string."""

    if result is None:
        return "(no result)"

    parts: List[str] = []
    structured = getattr(result, "structuredContent", None)
    content_blocks = getattr(result, "content", None)

    if content_blocks:
        for block in content_blocks:
            if isinstance(block, mcp_types.TextContent):
                parts.append(block.text)
            elif hasattr(block, "model_dump"):
                parts.append(json.dumps(block.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(block))

    if structured:
        try:
            parts.append(json.dumps(structured, ensure_ascii=False))
        except TypeError:
            parts.append(str(structured))

    if not parts:
        is_error = getattr(result, "isError", False)
        return "tool reported an error" if is_error else ""
    return "\n".join(part for part in parts if part)
