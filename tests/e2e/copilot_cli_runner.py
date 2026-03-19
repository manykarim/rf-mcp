"""Copilot CLI test runner for E2E testing against the rf-mcp MCP server.

Invokes the GitHub Copilot CLI via subprocess to run prompts against
the robotmcp MCP server, parses JSONL output, and returns structured
results with tool call tracking and usage metrics.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CopilotToolCall:
    """A single tool call extracted from Copilot CLI JSONL output."""

    tool_name: str
    mcp_tool_name: str
    mcp_server_name: str
    arguments: Dict[str, Any]
    tool_call_id: str = ""
    success: bool = True
    error: Optional[str] = None
    model: Optional[str] = None


@dataclass
class CopilotMessage:
    """An assistant message from Copilot CLI output."""

    content: str
    output_tokens: int = 0
    tool_requests: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CopilotRunResult:
    """Structured result of a Copilot CLI invocation."""

    model: str
    prompt: str
    tool_calls: List[CopilotToolCall] = field(default_factory=list)
    messages: List[CopilotMessage] = field(default_factory=list)
    exit_code: int = -1
    premium_requests: int = 0
    api_duration_ms: int = 0
    session_duration_ms: int = 0
    success: bool = False
    raw_events: List[Dict[str, Any]] = field(default_factory=list)
    stderr: str = ""

    def get_tool_names(self) -> List[str]:
        """Return list of MCP tool names that were called."""
        return [tc.mcp_tool_name for tc in self.tool_calls]

    def has_tool_call(self, tool_name: str) -> bool:
        """Check if a given MCP tool was called (by short name)."""
        return any(
            tc.mcp_tool_name == tool_name or tc.tool_name.endswith(f"-{tool_name}")
            for tc in self.tool_calls
        )

    def get_successful_tool_calls(self) -> List[CopilotToolCall]:
        """Return only successful tool calls."""
        return [tc for tc in self.tool_calls if tc.success]

    def get_failed_tool_calls(self) -> List[CopilotToolCall]:
        """Return only failed tool calls."""
        return [tc for tc in self.tool_calls if not tc.success]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


# Default MCP config template for CI environments.
# __PROJECT_DIR__ is replaced at runtime with the actual project path.
MCP_CONFIG_TEMPLATE = {
    "mcpServers": {
        "RobotMCP": {
            "type": "stdio",
            "command": "uv",
            "tools": ["*"],
            "args": [
                "run",
                "--directory",
                "__PROJECT_DIR__",
                "-m",
                "robotmcp.server",
            ],
            "env": {
                "ROBOTMCP_INSTRUCTIONS": "default",
                "ROBOTMCP_OUTPUT_MODE": "auto",
            },
        }
    }
}


def _find_copilot_binary() -> Optional[str]:
    """Find the copilot binary in PATH."""
    return shutil.which("copilot")


def is_copilot_available() -> bool:
    """Check whether the copilot CLI binary is available."""
    return _find_copilot_binary() is not None


def is_copilot_authenticated() -> bool:
    """Check whether Copilot authentication is available.

    Returns True if COPILOT_GITHUB_TOKEN is set or the copilot
    binary exists (implying local auth via GitHub CLI).
    """
    if os.environ.get("COPILOT_GITHUB_TOKEN"):
        return True
    return is_copilot_available()


def generate_mcp_config(project_dir: Optional[str] = None) -> str:
    """Generate an MCP config JSON file for CI and return its path.

    Args:
        project_dir: Project root directory. Defaults to rf-mcp repo root.

    Returns:
        Path to the generated temporary MCP config JSON file.
    """
    if project_dir is None:
        project_dir = str(Path(__file__).resolve().parent.parent.parent)

    config = json.loads(json.dumps(MCP_CONFIG_TEMPLATE))
    # Replace placeholder with actual path
    args = config["mcpServers"]["RobotMCP"]["args"]
    config["mcpServers"]["RobotMCP"]["args"] = [
        project_dir if a == "__PROJECT_DIR__" else a for a in args
    ]

    tmpfile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="copilot_mcp_", delete=False
    )
    json.dump(config, tmpfile, indent=2)
    tmpfile.close()
    return tmpfile.name


def _parse_jsonl_output(raw_output: str) -> List[Dict[str, Any]]:
    """Parse JSONL output from Copilot CLI, filtering ephemeral streaming deltas.

    Args:
        raw_output: Raw stdout from the copilot process.

    Returns:
        List of parsed JSON event dicts (non-ephemeral only).
    """
    events: List[Dict[str, Any]] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Skip ephemeral streaming deltas
        if obj.get("ephemeral", False):
            continue
        events.append(obj)
    return events


def _extract_tool_calls(events: List[Dict[str, Any]]) -> List[CopilotToolCall]:
    """Extract tool calls from parsed events.

    Pairs tool.execution_start with tool.execution_complete events.
    """
    # Index start events by toolCallId
    pending: Dict[str, CopilotToolCall] = {}
    completed: List[CopilotToolCall] = []

    for evt in events:
        evt_type = evt.get("type", "")
        data = evt.get("data", {})

        if evt_type == "tool.execution_start":
            tool_name = data.get("toolName", "")
            mcp_server = data.get("mcpServerName", "")
            mcp_tool = data.get("mcpToolName", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}

            call_id = data.get("toolCallId", evt.get("id", tool_name))

            tc = CopilotToolCall(
                tool_name=tool_name,
                mcp_tool_name=mcp_tool or _extract_mcp_tool_name(tool_name),
                mcp_server_name=mcp_server,
                arguments=arguments,
                tool_call_id=call_id,
            )
            pending[call_id] = tc

        elif evt_type == "tool.execution_complete":
            call_id = data.get("toolCallId", "")
            success = data.get("success", True)
            error_obj = data.get("error")
            error = error_obj.get("message") if isinstance(error_obj, dict) else error_obj
            model = data.get("model")

            if call_id in pending:
                tc = pending.pop(call_id)
                tc.success = success
                tc.error = error
                tc.model = model
                completed.append(tc)
            else:
                # Completion without matching start — still record it
                completed.append(
                    CopilotToolCall(
                        tool_name=call_id,
                        mcp_tool_name=_extract_mcp_tool_name(call_id),
                        mcp_server_name="",
                        arguments={},
                        tool_call_id=call_id,
                        success=success,
                        error=error,
                        model=model,
                    )
                )

    # Any remaining pending starts without completion are assumed successful
    for tc in pending.values():
        completed.append(tc)

    return completed


def _extract_mcp_tool_name(full_name: str) -> str:
    """Extract the short MCP tool name from a prefixed tool name.

    E.g. 'robotmcp-find_keywords' -> 'find_keywords'
         'RobotMCP-execute_step'  -> 'execute_step'
         'find_keywords'          -> 'find_keywords'
    """
    for prefix in ("robotmcp-", "RobotMCP-", "robotmcp_", "RobotMCP_"):
        if full_name.startswith(prefix):
            return full_name[len(prefix):]
    # Also handle dot-separated: robotmcp.find_keywords
    if "." in full_name:
        return full_name.split(".")[-1]
    return full_name


def _extract_messages(events: List[Dict[str, Any]]) -> List[CopilotMessage]:
    """Extract assistant messages from events."""
    messages: List[CopilotMessage] = []
    for evt in events:
        if evt.get("type") == "assistant.message":
            data = evt.get("data", {})
            content = data.get("content", "")
            output_tokens = data.get("outputTokens", 0)
            tool_requests = data.get("toolRequests", [])
            messages.append(
                CopilotMessage(
                    content=content,
                    output_tokens=output_tokens,
                    tool_requests=tool_requests,
                )
            )
    return messages


def _extract_result_metrics(
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract final result event metrics."""
    for evt in events:
        if evt.get("type") == "result":
            usage = evt.get("usage", {})
            return {
                "exit_code": evt.get("exitCode", 0),
                "premium_requests": usage.get("premiumRequests", 0),
                "api_duration_ms": usage.get("totalApiDurationMs", 0),
                "session_duration_ms": usage.get("sessionDurationMs", 0),
            }
    return {}


def run_copilot_cli(
    prompt: str,
    model: str = "gpt-5-mini",
    timeout: int = 180,
    mcp_config_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> CopilotRunResult:
    """Run the Copilot CLI with the given prompt and return structured results.

    Args:
        prompt: The prompt text to send to Copilot.
        model: Model name to use (e.g. 'gpt-5-mini', 'claude-haiku-4.5').
        timeout: Timeout in seconds for the subprocess.
        mcp_config_path: Optional path to an MCP config JSON file.
            Pass ``"auto"`` to generate one (used in CI).  If ``None``,
            no additional config is passed — the copilot CLI will use its
            existing local config (``~/.copilot/mcp-config.json``).
        extra_args: Optional additional CLI arguments.

    Returns:
        CopilotRunResult with parsed tool calls, messages, and metrics.

    Raises:
        FileNotFoundError: If the copilot binary is not in PATH.
    """
    copilot_bin = _find_copilot_binary()
    if copilot_bin is None:
        raise FileNotFoundError("copilot binary not found in PATH")

    # Generate MCP config only in CI (when explicitly requested)
    generated_config = False
    if mcp_config_path == "auto":
        mcp_config_path = generate_mcp_config()
        generated_config = True

    try:
        cmd = [
            copilot_bin,
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "json",
            "--no-ask-user",
            "--allow-tool=RobotMCP",
            "--allow-tool=robotmcp",
            # Disable non-essential MCP servers to reduce startup time
            "--disable-builtin-mcps",
            "--disable-mcp-server=claude-flow",
        ]

        # Only add MCP config when explicitly provided (CI environments)
        if mcp_config_path:
            cmd.extend(["--additional-mcp-config", f"@{mcp_config_path}"])

        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired as exc:
            # Parse whatever output was produced before timeout
            raw = (exc.stdout or b"").decode("utf-8", errors="replace")
            events = _parse_jsonl_output(raw)
            return CopilotRunResult(
                model=model,
                prompt=prompt,
                tool_calls=_extract_tool_calls(events),
                messages=_extract_messages(events),
                exit_code=-1,
                success=False,
                raw_events=events,
                stderr=f"Timeout after {timeout}s",
            )

        # Parse events from stdout
        events = _parse_jsonl_output(result.stdout)
        tool_calls = _extract_tool_calls(events)
        messages = _extract_messages(events)
        metrics = _extract_result_metrics(events)

        return CopilotRunResult(
            model=model,
            prompt=prompt,
            tool_calls=tool_calls,
            messages=messages,
            exit_code=metrics.get("exit_code", result.returncode),
            premium_requests=metrics.get("premium_requests", 0),
            api_duration_ms=metrics.get("api_duration_ms", 0),
            session_duration_ms=metrics.get("session_duration_ms", 0),
            success=result.returncode == 0,
            raw_events=events,
            stderr=result.stderr,
        )

    finally:
        # Clean up generated config
        if generated_config and mcp_config_path and os.path.exists(mcp_config_path):
            os.unlink(mcp_config_path)
