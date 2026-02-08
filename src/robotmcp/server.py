"""Main MCP Server implementation for Robot Framework integration."""

import argparse
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from fastmcp import Context, FastMCP

from robotmcp.components.execution import ExecutionCoordinator
from robotmcp.components.execution.external_rf_client import ExternalRFClient
from robotmcp.components.execution.mobile_capability_service import (
    MobileCapabilityService,
)
from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)
from robotmcp.components.keyword_matcher import KeywordMatcher
from robotmcp.components.library_recommender import LibraryRecommender
from robotmcp.components.nlp_processor import NaturalLanguageProcessor
from robotmcp.components.state_manager import StateManager
from robotmcp.components.test_builder import TestBuilder
from robotmcp.config import library_registry
from robotmcp.domains.instruction import FastMCPInstructionAdapter
from robotmcp.models.session_models import PlatformType
from robotmcp.optimization.instruction_hooks import InstructionLearningHooks
from robotmcp.plugins import get_library_plugin_manager
from robotmcp.utils.server_integration import initialize_enhanced_serialization

logger = logging.getLogger(__name__)

# Initialize instruction learning hooks singleton
_instruction_hooks: Optional[InstructionLearningHooks] = None


def _get_instruction_hooks() -> InstructionLearningHooks:
    """Get or initialize the instruction learning hooks.

    Returns:
        The singleton InstructionLearningHooks instance
    """
    global _instruction_hooks
    if _instruction_hooks is None:
        _instruction_hooks = InstructionLearningHooks.get_instance()
    return _instruction_hooks


def _track_tool_result(
    session_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    """Track a tool call result for instruction learning.

    Args:
        session_id: Session identifier
        tool_name: Name of the tool that was called
        arguments: Arguments passed to the tool
        result: Result dictionary from the tool
    """
    try:
        hooks = _get_instruction_hooks()
        success = result.get("success", True)
        error = result.get("error") if not success else None
        hooks.on_tool_call(session_id, tool_name, arguments, success, error, result)
    except Exception as e:
        # Don't let learning errors affect tool execution
        logger.debug(f"Instruction learning tracking error: {e}")


if TYPE_CHECKING:
    from robotmcp.frontend.controller import FrontendServerController


def _resolve_server_instructions() -> Optional[str]:
    """Resolve MCP instructions based on environment configuration.

    Environment variables (as per ADR-002):
    - ROBOTMCP_INSTRUCTIONS: Mode control ("off", "default", "custom")
    - ROBOTMCP_INSTRUCTIONS_TEMPLATE: Template selection
      ("minimal", "standard", "detailed", "browser-focused", "api-focused")
    - ROBOTMCP_INSTRUCTIONS_FILE: Path to custom instructions file

    Returns:
        Instruction string for FastMCP, or None if instructions are disabled.
    """
    try:
        # Create adapter and resolve configuration from environment
        adapter = FastMCPInstructionAdapter()
        config = adapter.create_config_from_env()

        # Get instructions formatted for FastMCP
        instructions = adapter.get_server_instructions(
            config,
            context=adapter.get_default_tools_context(),
        )

        if instructions:
            logger.info(
                "MCP instructions enabled: mode=%s, template=%s, length=%d chars",
                config.mode.value,
                adapter.get_template_type().value if config.is_enabled else "n/a",
                len(instructions),
            )
        else:
            logger.info("MCP instructions disabled")

        return instructions

    except Exception as e:
        logger.warning(
            "Failed to resolve MCP instructions, continuing without: %s",
            str(e),
        )
        return None


def _create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server with instructions.

    Returns:
        Configured FastMCP server instance.
    """
    instructions = _resolve_server_instructions()

    return FastMCP(
        "Robot Framework MCP Server",
        instructions=instructions,
    )


# Initialize FastMCP server with instructions
mcp = _create_mcp_server()

# Optional reference to the running frontend controller
_frontend_controller: "FrontendServerController | None" = None


def _get_external_client_if_configured() -> ExternalRFClient | None:
    """Return an ExternalRFClient when attach mode is configured AND reachable.

    Uses a fast cached TCP probe (0.3ms when refused, 5s cache TTL) so callers
    never block on an unreachable bridge.

    Env vars:
    - ROBOTMCP_ATTACH_HOST (required to enable attach mode)
    - ROBOTMCP_ATTACH_PORT (optional, defaults 7317)
    - ROBOTMCP_ATTACH_TOKEN (optional, defaults 'change-me')
    """
    try:
        host = os.environ.get("ROBOTMCP_ATTACH_HOST")
        if not host:
            return None
        mode = os.environ.get("ROBOTMCP_ATTACH_DEFAULT", "auto").strip().lower()
        if mode == "off":
            return None
        port = int(os.environ.get("ROBOTMCP_ATTACH_PORT", "7317"))
        token = os.environ.get("ROBOTMCP_ATTACH_TOKEN", "change-me")
        client = ExternalRFClient(host=host, port=port, token=token)
        # Fast probe: if bridge port is unreachable, return None immediately
        if not client.is_reachable():
            return None
        return client
    except Exception:
        return None


def _call_attach_tool_with_fallback(
    tool_name: str,
    external_call: Callable[[ExternalRFClient], Dict[str, Any]],
    local_call: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute an attach-aware tool with automatic fallback when bridge is unreachable."""

    client = _get_external_client_if_configured()
    mode = os.environ.get("ROBOTMCP_ATTACH_DEFAULT", "auto").strip().lower()
    strict = os.environ.get("ROBOTMCP_ATTACH_STRICT", "0").strip() in {
        "1",
        "true",
        "yes",
    }

    if client is None or mode == "off":
        return local_call()

    try:
        response = external_call(client)
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive conversion to attach-style error
        err = str(exc)
        logger.error(
            "ATTACH tool '%s' raised exception: %s", tool_name, err, exc_info=False
        )
        response = {"success": False, "error": err}

    if response.get("success"):
        return response

    error_msg = response.get("error", "attach call failed")
    logger.error("ATTACH tool '%s' error: %s", tool_name, error_msg)

    if strict or mode == "force":
        return {
            "success": False,
            "error": f"Attach bridge call failed ({tool_name}): {error_msg}",
        }

    logger.warning(
        "ATTACH unreachable for '%s'; falling back to local execution", tool_name
    )
    return local_call()


def _frontend_dependencies_available() -> bool:
    """Check whether optional frontend dependencies are installed."""

    try:
        import django  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        return False
    return True


def _install_frontend_lifespan(config: "FrontendConfig") -> None:
    """Attach a custom FastMCP lifespan that manages the Django frontend."""

    from robotmcp.frontend.controller import FrontendServerController

    global _frontend_controller

    controller = FrontendServerController(config)

    @asynccontextmanager
    async def frontend_lifespan(server: FastMCP):  # type: ignore[override]
        try:
            await controller.start()
            yield {"frontend_url": controller.url}
        finally:
            await controller.stop()

    mcp._mcp_server.lifespan = frontend_lifespan  # type: ignore[attr-defined]
    _frontend_controller = controller
    logger.info(
        "Frontend enabled at http://%s:%s%s",
        config.host,
        config.port,
        config.base_path,
    )


def _install_health_monitor_lifespan(
    interval_seconds: int = 60,
    failure_threshold: int = 3,
) -> None:
    """Attach a FastMCP lifespan that manages the bridge health monitor.

    This function wraps any existing lifespan (e.g., frontend lifespan) to
    also start/stop the health monitor.

    Args:
        interval_seconds: Time between health checks.
        failure_threshold: Number of consecutive failures before cleanup.
    """
    global _health_monitor

    monitor = BridgeHealthMonitor(
        interval_seconds=interval_seconds,
        failure_threshold=failure_threshold,
    )
    _health_monitor = monitor

    # Capture existing lifespan (if any) to chain with it
    existing_lifespan = getattr(mcp._mcp_server, "lifespan", None)

    @asynccontextmanager
    async def health_monitor_lifespan(server: FastMCP):  # type: ignore[override]
        try:
            await monitor.start()

            # If there's an existing lifespan, chain with it
            if existing_lifespan is not None:
                async with existing_lifespan(server) as context:
                    yield context
            else:
                yield {}
        finally:
            await monitor.stop()

    mcp._mcp_server.lifespan = health_monitor_lifespan  # type: ignore[attr-defined]


def _log_attach_banner() -> None:
    """Log attach-mode configuration and basic bridge health at server start."""

    # Log several environment variables for debugging
    logger.info(
        (
            "--- RobotMCP Environment Variables ---\n"
            f"ROBOTMCP_ATTACH_HOST: {os.environ.get('ROBOTMCP_ATTACH_HOST')}\n"
            f"ROBOTMCP_ATTACH_PORT: {os.environ.get('ROBOTMCP_ATTACH_PORT')}\n"
            f"ROBOTMCP_ATTACH_TOKEN: {os.environ.get('ROBOTMCP_ATTACH_TOKEN')}\n"
        )
    )
    try:
        client = _get_external_client_if_configured()
        if client is None:
            logger.info("Attach mode: disabled (ROBOTMCP_ATTACH_HOST not set)")
            return
        logger.info(f"Attach mode: enabled → {client.host}:{client.port}")
        diag = client.diagnostics()
        if diag.get("success"):
            details = diag.get("result") or {}
            libs = details.get("libraries")
            extra = f" libraries={libs}" if libs else ""
            logger.info(f"Attach bridge: reachable.{extra}")
        else:
            err = diag.get("error", "not reachable yet")
            logger.info(f"Attach bridge: not reachable ({err})")
        mode = os.environ.get("ROBOTMCP_ATTACH_DEFAULT", "auto").strip().lower()
        strict = os.environ.get("ROBOTMCP_ATTACH_STRICT", "0").strip() in {
            "1",
            "true",
            "yes",
        }
        logger.info(f"Attach default: {mode}{' (strict)' if strict else ''}")
    except Exception as e:  # defensive
        logger.info(f"Attach bridge: check failed ({e})")


def _startup_bridge_validation() -> Dict[str, Any]:
    """Validate bridge at startup and reset any stale local state.

    This function implements Phase 1 of ADR-004 Debug Bridge Connection Cleanup:
    - Checks bridge health on MCP server startup
    - Cleans local session state to synchronize with bridge state
    - Configurable via ROBOTMCP_STARTUP_CLEANUP environment variable

    Environment variable ROBOTMCP_STARTUP_CLEANUP:
    - "auto" (default): Clean sessions only if bridge is healthy and context is active
    - "always": Always clean local sessions at startup
    - "off": Disable startup cleanup entirely

    Returns:
        Dict with validation results and actions taken:
        - cleanup_mode: The configured cleanup mode
        - attach_mode: Whether attach mode is configured
        - healthy: Whether bridge is healthy (when attach mode enabled)
        - cleanup_performed: Whether cleanup was performed
        - sessions_cleaned: Number of sessions cleaned (if any)
        - error: Error message (if cleanup failed)
    """
    cleanup_mode = os.environ.get("ROBOTMCP_STARTUP_CLEANUP", "auto").lower().strip()

    # Validate cleanup mode
    if cleanup_mode not in ("auto", "always", "off"):
        logger.warning(
            f"Invalid ROBOTMCP_STARTUP_CLEANUP value '{cleanup_mode}', using 'auto'"
        )
        cleanup_mode = "auto"

    if cleanup_mode == "off":
        logger.info("Startup cleanup disabled via ROBOTMCP_STARTUP_CLEANUP=off")
        return {"cleanup_mode": "off", "skipped": True}

    client = _get_external_client_if_configured()
    if client is None:
        # No attach mode configured - nothing to validate
        logger.debug("Startup validation: attach mode not configured")
        return {"attach_mode": False, "cleanup_performed": False}

    try:
        # Check bridge health
        health = _check_bridge_health(client)

        if cleanup_mode == "auto":
            # In auto mode, only clean if bridge is healthy and context is active
            if health.get("healthy") and health.get("context_active"):
                cleaned = execution_engine.session_manager.cleanup_all_sessions()
                logger.info(
                    f"Startup cleanup: reset {cleaned} local sessions to sync with bridge"
                )
                return {
                    "cleanup_mode": "auto",
                    "attach_mode": True,
                    "healthy": True,
                    "context_active": True,
                    "cleanup_performed": True,
                    "sessions_cleaned": cleaned,
                }
            else:
                # Bridge not healthy or no active context - keep local sessions
                reason = (
                    "bridge not healthy"
                    if not health.get("healthy")
                    else "no active RF context"
                )
                logger.info(f"Startup cleanup: skipped ({reason}), keeping local sessions")
                return {
                    "cleanup_mode": "auto",
                    "attach_mode": True,
                    "healthy": health.get("healthy", False),
                    "context_active": health.get("context_active", False),
                    "cleanup_performed": False,
                    "reason": reason,
                }

        elif cleanup_mode == "always":
            # Always clean local sessions at startup, regardless of bridge health
            cleaned = execution_engine.session_manager.cleanup_all_sessions()
            logger.info(f"Startup cleanup (always mode): reset {cleaned} local sessions")
            return {
                "cleanup_mode": "always",
                "attach_mode": True,
                "cleanup_performed": True,
                "sessions_cleaned": cleaned,
                "bridge_healthy": health.get("healthy", False),
                "context_active": health.get("context_active", False),
            }

    except Exception as e:
        logger.warning(f"Startup validation failed: {e}")
        return {
            "cleanup_mode": cleanup_mode,
            "attach_mode": True,
            "error": str(e),
            "cleanup_performed": False,
        }

    # Should not reach here, but return a safe default
    return {"cleanup_mode": cleanup_mode, "attach_mode": True, "cleanup_performed": False}


def _compute_effective_use_context(
    use_context: bool | None, client: ExternalRFClient | None, keyword: str
) -> tuple[bool, str, bool]:
    """Decide whether to route to the external bridge.

    Returns a tuple: (effective_use_context, mode, strict)
    - mode: value of ROBOTMCP_ATTACH_DEFAULT (auto|force|off)
    - strict: True if ROBOTMCP_ATTACH_STRICT is enabled
    """
    mode = os.environ.get("ROBOTMCP_ATTACH_DEFAULT", "auto").strip().lower()
    strict = os.environ.get("ROBOTMCP_ATTACH_STRICT", "0").strip() in {
        "1",
        "true",
        "yes",
    }
    effective = bool(use_context) if use_context is not None else False
    if client is not None:
        if use_context is None:
            if mode in ("auto", "force"):
                reachable = bool(client.diagnostics().get("success"))
                if mode == "force" or reachable:
                    effective = True
                    logger.info(
                        f"ATTACH mode ({mode}): defaulting use_context=True for '{keyword}'"
                    )
                else:
                    effective = False
                    logger.info(
                        f"ATTACH mode (auto): bridge unreachable, defaulting to local for '{keyword}'"
                    )
        elif use_context is False and mode == "force":
            effective = True
            logger.info(
                f"ATTACH mode (force): overriding use_context=False → True for '{keyword}'"
            )
    return effective, mode, strict


def _get_external_client_with_session_sync(
    session_id: str = "default",
) -> tuple[ExternalRFClient | None, "ExecutionSession | None"]:
    """Get bridge client and ensure local session is synchronized.

    When bridge is configured and reachable with an active RF context,
    automatically creates the local session if it doesn't exist.

    This function implements Phase 1 of ADR-003 Debug Bridge Unification:
    - S1: Auto-Initialize Session from Bridge
    - Fixes the get_session_state first-call failure when bridge is active

    Returns:
        (client, session) tuple where:
        - client is the ExternalRFClient if bridge is configured and reachable
        - session is the local ExecutionSession (may be auto-created)
        Both may be None if bridge not configured and no local session exists.
    """
    client = _get_external_client_if_configured()
    session = execution_engine.session_manager.get_session(session_id)

    if client is None:
        # No bridge - return existing local session (may be None)
        return None, session

    # Bridge configured - check reachability and context
    try:
        diag = client.diagnostics()
        if not diag.get("success"):
            # Bridge unreachable - fall back to local
            return None, session

        # Bridge reachable with context - ensure local session exists
        if session is None:
            bridge_result = diag.get("result", {})
            if bridge_result.get("context", False):
                # Auto-create session from bridge state
                session = execution_engine.session_manager.create_session(session_id)
                for lib in bridge_result.get("libraries", []):
                    if lib not in session.imported_libraries:
                        session.imported_libraries.append(lib)
                    session.loaded_libraries.add(lib)
                if bridge_result.get("libraries"):
                    # Preserve BuiltIn at front, add bridge libraries after
                    bridge_libs = [
                        lib
                        for lib in bridge_result.get("libraries", [])
                        if lib != "BuiltIn"
                    ]
                    # Keep BuiltIn first if present in search_order
                    if session.search_order and session.search_order[0] == "BuiltIn":
                        session.search_order = ["BuiltIn"] + bridge_libs
                    else:
                        session.search_order = bridge_libs
                logger.info(
                    f"Auto-created session '{session_id}' from bridge with "
                    f"libraries: {bridge_result.get('libraries', [])}"
                )

        return client, session

    except Exception as e:
        logger.debug(f"Bridge check failed: {e}")
        return None, session


async def _sync_session_bidirectional(
    session_id: str,
    client: ExternalRFClient,
    session: "ExecutionSession",
    direction: str = "both",  # "to_bridge", "from_bridge", "both"
) -> Dict[str, Any]:
    """Synchronize session state between local and bridge.

    This function implements Phase 4 of ADR-003 Debug Bridge Unification:
    - Bidirectional sync of libraries and variables
    - Ensures local session stays in sync with live RF process

    Args:
        session_id: Session identifier for logging
        client: ExternalRFClient connected to the bridge
        session: Local ExecutionSession to sync
        direction: Sync direction - "to_bridge", "from_bridge", or "both"

    Returns:
        Dict with success status and list of synced items
    """
    results: Dict[str, Any] = {"success": True, "synced": [], "session_id": session_id}

    try:
        if direction in ("from_bridge", "both"):
            # Sync libraries from bridge
            diag = client.diagnostics()
            if diag.get("success"):
                bridge_libs = set(diag.get("result", {}).get("libraries", []))
                for lib in bridge_libs:
                    if lib not in session.imported_libraries:
                        session.imported_libraries.append(lib)
                    session.loaded_libraries.add(lib)
                results["synced"].append("libraries_from_bridge")
                results["libraries_synced"] = list(bridge_libs)

            # Sync variables from bridge
            var_resp = client.get_variables()
            if var_resp.get("success"):
                raw_vars = var_resp.get("result", {})
                synced_count = 0
                for k, v in raw_vars.items():
                    # Strip ${} wrappers from variable names
                    clean_key = k
                    if clean_key.startswith("${") and clean_key.endswith("}"):
                        clean_key = clean_key[2:-1]
                    session.variables[clean_key] = v
                    synced_count += 1
                results["synced"].append("variables_from_bridge")
                results["variables_synced_count"] = synced_count

        if direction in ("to_bridge", "both"):
            # Push local variables to bridge
            pushed_count = 0
            push_errors = []
            for key, value in session.variables.items():
                try:
                    resp = client.set_variable(key, value, scope="suite")
                    if resp.get("success"):
                        pushed_count += 1
                    else:
                        push_errors.append(f"{key}: {resp.get('error', 'unknown')}")
                except Exception as var_err:
                    push_errors.append(f"{key}: {var_err}")
            results["synced"].append("variables_to_bridge")
            results["variables_pushed_count"] = pushed_count
            if push_errors:
                results["push_errors"] = push_errors[:5]  # Limit error list

        return results

    except Exception as e:
        logger.error(f"Session bidirectional sync failed: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


def _check_bridge_health(client: ExternalRFClient) -> Dict[str, Any]:
    """Check bridge health and provide recovery hints.

    This function implements Phase 4 of ADR-003 Debug Bridge Unification:
    - Health check for the attach bridge
    - Provides actionable recovery hints for common errors

    Args:
        client: ExternalRFClient to check

    Returns:
        Dict with health status and recovery hints:
        - healthy: True if bridge is reachable and has active context
        - context_active: True if RF execution context is active
        - error: Error type if unhealthy
        - recovery_hint: Actionable hint for recovery
    """
    try:
        diag = client.diagnostics()
        if diag.get("success"):
            bridge_result = diag.get("result", {})
            return {
                "healthy": True,
                "context_active": bridge_result.get("context", False),
                "libraries": bridge_result.get("libraries", []),
                "host": client.host,
                "port": client.port,
            }
        # Bridge responded but returned error
        return {
            "healthy": False,
            "context_active": False,
            "error": "diagnostics_failed",
            "details": diag.get("error", "unknown error"),
            "recovery_hint": "Bridge responded but returned error. Check RF process logs for details.",
        }
    except Exception as e:
        error_str = str(e).lower()

        if "connection refused" in error_str:
            return {
                "healthy": False,
                "context_active": False,
                "error": "connection_refused",
                "recovery_hint": (
                    "Bridge server not running. Ensure Robot Framework is running "
                    "with McpAttach library loaded and 'MCP Serve' keyword has been called."
                ),
            }
        elif "timeout" in error_str or "timed out" in error_str:
            return {
                "healthy": False,
                "context_active": False,
                "error": "timeout",
                "recovery_hint": (
                    "Bridge server not responding. RF process may be blocked, "
                    "overloaded, or executing a long-running keyword. "
                    "Try again after the current operation completes."
                ),
            }
        elif (
            "name or service not known" in error_str
            or "nodename nor servname" in error_str
        ):
            return {
                "healthy": False,
                "context_active": False,
                "error": "dns_resolution_failed",
                "recovery_hint": (
                    f"Cannot resolve bridge host. Check ROBOTMCP_ATTACH_HOST "
                    f"environment variable (current: {client.host})."
                ),
            }
        elif "network is unreachable" in error_str:
            return {
                "healthy": False,
                "context_active": False,
                "error": "network_unreachable",
                "recovery_hint": (
                    "Network is unreachable. Check network connectivity "
                    "and firewall settings."
                ),
            }

        return {
            "healthy": False,
            "context_active": False,
            "error": str(e),
            "recovery_hint": (
                "Unknown error connecting to bridge. Check network connectivity, "
                "RF process status, and environment variables "
                "(ROBOTMCP_ATTACH_HOST, ROBOTMCP_ATTACH_PORT)."
            ),
        }


# Internal helpers to build prompt texts (used by both @mcp.prompt and wrapper tools)
def _build_recommend_libraries_sampling_prompt(
    scenario: str,
    k: int = 4,
    available_libraries: List[Dict[str, Any]] = None,
) -> str:
    try:
        import json

        libs_section = (
            json.dumps(available_libraries, ensure_ascii=False, indent=2)
            if available_libraries
            else "[]"
        )
    except Exception:
        libs_section = "[]"

    return (
        "# Task\n"
        "You are 1 of {k} samplers. Recommend the best Robot Framework libraries for this scenario.\n"
        "- Consider ONLY the libraries listed below as available in this environment.\n"
        "- Resolve conflicts (e.g., prefer one of Browser/SeleniumLibrary).\n"
        "- Output strictly the JSON schema in the Output Format section.\n\n"
        "# Scenario\n"
        f"{scenario}\n\n"
        "# Available Libraries (from environment)\n"
        f"{libs_section}\n\n"
        "# Guidance\n"
        "- Choose 2–5 libraries maximum.\n"
        "- Justify each choice concisely, referencing capabilities from 'available_libraries'.\n"
        "- If multiple web libs exist, pick one with a short rationale.\n"
        "- For API use, mention RequestsLibrary and how sessions are created.\n"
        "- For XML/data flows, consider XML/Collections/String.\n"
        "- If specialized libs are not needed, do not recommend them.\n\n"
        "# Output Format (JSON)\n"
        "{\n"
        '  "recommendations": [\n'
        '    { "name": "<LibraryName>", "reason": "<1-2 lines>", "score": 0.0 },\n'
        "    ... up to 5 total ...\n"
        "  ],\n"
        '  "conflicts": [\n'
        '    { "conflict_set": ["Browser", "SeleniumLibrary"], "chosen": "Browser", "reason": "<1 line>" }\n'
        "  ]\n"
        "}\n"
    )


def _build_choose_recommendations_prompt(
    candidates: List[Dict[str, Any]] = None,
) -> str:
    import json

    cand_section = (
        json.dumps(candidates, ensure_ascii=False, indent=2) if candidates else "[]"
    )

    return (
        "# Task\n"
        "Select or merge the following sampled recommendations into a final JSON.\n"
        "- Deduplicate libraries by name.\n"
        "- Resolve conflicts (e.g., Browser vs SeleniumLibrary) by choosing the higher total score; state a 1-line reason.\n"
        "- Normalize scores to 0..1, and keep at most 5 libraries.\n"
        "- Output strictly the JSON under 'Output Format'.\n\n"
        "# Candidates (JSON)\n"
        f"{cand_section}\n\n"
        "# Output Format (JSON)\n"
        "{\n"
        '  "recommendations": [\n'
        '    { "name": "<LibraryName>", "reason": "<1-2 lines>", "score": 0.0 }\n'
        "  ],\n"
        '  "conflicts": [\n'
        '    { "conflict_set": ["Browser", "SeleniumLibrary"], "chosen": "<name>", "reason": "<1 line>" }\n'
        "  ]\n"
        "}\n"
    )


# Initialize components
nlp_processor = NaturalLanguageProcessor()
keyword_matcher = KeywordMatcher()
library_recommender = LibraryRecommender()
execution_engine = ExecutionCoordinator()
state_manager = StateManager()
test_builder = TestBuilder(execution_engine)
mobile_capability_service = MobileCapabilityService()

# Initialize enhanced serialization system
initialize_enhanced_serialization(execution_engine)

# Shared guidance for automation workflows
AUTOMATION_TOOL_GUIDE: List[tuple[str, str]] = [
    (
        "analyze_scenario",
        "to understand the requirements and create/configure a session.",
    ),
    (
        "recommend_libraries",
        "to fetch targeted library suggestions for the scenario.",
    ),
    (
        "execute_step",
        "to run individual keywords in the active session (reuse the same session_id).",
    ),
    (
        "get_session_state",
        "to capture application state, DOM snapshots, screenshots, and variables when debugging.",
    ),
    (
        "diagnose_rf_context",
        "to inspect the Robot Framework namespace (libraries, variables, search order) if keywords fail.",
    ),
    (
        "build_test_suite",
        "to compile the validated steps into a reusable Robot Framework suite.",
    ),
    (
        "run_test_suite_dry",
        "to perform a staged dry run and validate the generated suite structure.",
    ),
    (
        "run_test_suite",
        "to execute the finalized suite with all required libraries loaded.",
    ),
]


# Helper functions
async def _ensure_all_session_libraries_loaded():
    """
    Ensure all imported session libraries are loaded in LibraryManager.

    Enhanced validation to prevent keyword filtering issues and provide better error reporting.
    """
    try:
        session_manager = execution_engine.session_manager
        all_sessions = session_manager.sessions.values()

        for session in all_sessions:
            for library_name in session.imported_libraries:
                # Check if library is loaded in the orchestrator
                if library_name not in execution_engine.keyword_discovery.libraries:
                    logger.warning(
                        f"Session library '{library_name}' not loaded in orchestrator, attempting to load"
                    )
                    session._ensure_library_loaded_immediately(library_name)

                    # Verify loading succeeded
                    if library_name not in execution_engine.keyword_discovery.libraries:
                        logger.error(
                            f"Failed to load session library '{library_name}' - may cause keyword filtering issues"
                        )
                else:
                    logger.debug(
                        f"Session library '{library_name}' already loaded in orchestrator"
                    )

        logger.debug(
            "Validated all session libraries are loaded for discovery operations"
        )

    except Exception as e:
        logger.error(f"Error ensuring session libraries loaded: {e}")
        # Don't fail the discovery operation, but log the issue for debugging


@mcp.prompt
def automate(scenario: str) -> str:
    """Uses RobotMCP to create a test suite from a scenario description"""
    tool_lines = "\n".join(
        f"{idx}. Use {tool} {description}"
        for idx, (tool, description) in enumerate(AUTOMATION_TOOL_GUIDE, start=1)
    )
    return (
        "# Task\n"
        "Use RobotMCP to create a TestSuite and execute it step wise.\n"
        f"{tool_lines}\n"
        "General hints:\n"
        "- For UI testing capture state via get_session_state (sections=['application_state','page_source','variables']).\n"
        "- Ensure Browser or Playwright contexts run in non-headless mode when interacting with live UIs.\n"
        "- When you need keyword or library details, use get_keyword_info (mode='library' or 'keyword') and get_library_documentation.\n"
        "- Use manage_session (set_variables/import_library) to configure sessions between steps if needed.\n"
        "# Scenario:\n"
        f"{scenario}\n"
    )


@mcp.prompt
def learn(scenario: str) -> str:
    """Guides a user through automation and explains the generated code/choices."""
    return (
        "# Role\n"
        "Act as a friendly Robot Framework tutor. Automate the scenario with RobotMCP tools, "
        "but after each major phase summarize what you did and why (libraries chosen, keywords executed, "
        "variables used, etc.). Keep explanations concise and practical.\n"
        "# Workflow\n"
        "1. analyze_scenario – understand requirements and capture the session_id.\n"
        "2. recommend_libraries – justify which libraries fit the scenario.\n"
        "3. manage_session / execute_step – build the test step-by-step, reusing the same session.\n"
        "4. get_session_state or diagnose_rf_context when you need to inspect UI/variables/libraries.\n"
        "5. build_test_suite – convert the validated steps into a suite.\n"
        "6. run_test_suite_dry (optional) – confirm the suite compiles.\n"
        "7. run_test_suite – execute if appropriate.\n"
        "# Teaching Guidance\n"
        "- Explain why each library/keyword was selected (e.g., Browser vs SeleniumLibrary, Images vs API).\n"
        "- Highlight any tricky locators, variables, or context setup.\n"
        "- Encourage best practices (non-headless browser, reusable keywords, variable naming) without lecturing.\n"
        "- Keep explanations short (2–3 sentences) and actionable.\n"
        "# Scenario\n"
        f"{scenario}\n"
    )


# Note: Prompt endpoints removed per Option B. Use tools below that return plain prompt text.


@mcp.tool(
    name="list_library_plugins",
    description="List discovered library plugins with basic metadata.",
    enabled=False,
)
async def list_library_plugins() -> Dict[str, Any]:
    """Return a summary of every loaded library plugin."""

    library_registry.get_all_libraries()
    manager = get_library_plugin_manager()

    plugins: List[Dict[str, Any]] = []
    for name in manager.list_plugin_names():
        metadata = manager.get_metadata(name)
        if not metadata:
            continue
        plugins.append(
            {
                "name": metadata.name,
                "package_name": metadata.package_name,
                "import_path": metadata.import_path,
                "library_type": metadata.library_type,
                "load_priority": metadata.load_priority,
                "source": manager.get_plugin_source(name) or "unknown",
                "default_enabled": metadata.default_enabled,
            }
        )

    return {"success": True, "plugins": plugins, "count": len(plugins)}


@mcp.tool(
    name="reload_library_plugins",
    description="Reload library plugins from builtin definitions, entry points, and manifests.",
    enabled=False,
)
async def reload_library_plugins_tool(
    manifest_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reload library plugins and return the resulting library list."""

    snapshot = library_registry.reload_library_plugins(manifest_paths)
    return {
        "success": True,
        "libraries": sorted(snapshot.keys()),
        "count": len(snapshot),
    }


@mcp.tool(
    name="diagnose_library_plugin",
    description="Inspect metadata, capabilities, and hooks for a specific library plugin.",
    enabled=False,
)
async def diagnose_library_plugin(plugin_name: str) -> Dict[str, Any]:
    """Return detailed information about a specific library plugin."""

    library_registry.get_all_libraries()
    manager = get_library_plugin_manager()

    metadata = manager.get_metadata(plugin_name)
    if not metadata:
        return {
            "success": False,
            "error": f"Plugin '{plugin_name}' not found.",
        }

    capabilities = manager.get_capabilities(plugin_name)
    install_actions = [
        {"description": action.description, "command": list(action.command)}
        for action in manager.get_install_actions(plugin_name)
    ]
    hints = manager.get_hints(plugin_name)
    prompts = manager.get_prompt_bundle(plugin_name)

    return {
        "success": True,
        "metadata": {
            "name": metadata.name,
            "package_name": metadata.package_name,
            "import_path": metadata.import_path,
            "library_type": metadata.library_type,
            "description": metadata.description,
            "use_cases": metadata.use_cases,
            "categories": metadata.categories,
            "contexts": metadata.contexts,
            "installation_command": metadata.installation_command,
            "post_install_commands": metadata.post_install_commands,
            "platform_requirements": metadata.platform_requirements,
            "dependencies": metadata.dependencies,
            "load_priority": metadata.load_priority,
            "default_enabled": metadata.default_enabled,
            "requires_type_conversion": metadata.requires_type_conversion,
            "supports_async": metadata.supports_async,
            "is_deprecated": metadata.is_deprecated,
            "extra_name": metadata.extra_name,
        },
        "capabilities": {
            "contexts": capabilities.contexts if capabilities else [],
            "features": capabilities.features if capabilities else [],
            "technology": capabilities.technology if capabilities else [],
            "supports_page_source": capabilities.supports_page_source
            if capabilities
            else False,
            "supports_application_state": capabilities.supports_application_state
            if capabilities
            else False,
            "requires_type_conversion": capabilities.requires_type_conversion
            if capabilities
            else False,
            "supports_async": capabilities.supports_async if capabilities else False,
        },
        "install_actions": install_actions,
        "hints": {
            "standard_keywords": hints.standard_keywords if hints else [],
            "error_hints": hints.error_hints if hints else [],
            "usage_examples": hints.usage_examples if hints else [],
        },
        "prompt_bundle": {
            "recommendation": prompts.recommendation if prompts else None,
            "troubleshooting": prompts.troubleshooting if prompts else None,
            "sampling_notes": prompts.sampling_notes if prompts else None,
        },
        "source": manager.get_plugin_source(plugin_name) or "unknown",
    }


@mcp.tool
async def manage_library_plugins(
    action: str = "list", plugin_name: str | None = None
) -> Dict[str, Any]:
    """Inspect or reload library plugins.

    Args:
        action: One of "list", "reload", or "diagnose".
        plugin_name: Plugin name when action="diagnose".

    Returns:
        Dict[str, Any]: Plugin metadata depending on action:
            - success: bool
            - action: echo of the requested action
            - plugins/plugin/reload_result: action-specific data
            - error: present on failure
    """

    action_norm = (action or "list").strip().lower()
    manager = get_library_plugin_manager()

    def _plugin_payload(name: str) -> Dict[str, Any]:
        metadata = manager.get_metadata(name)
        plugin = manager.get_plugin(name)
        capabilities = manager.get_capabilities(name)
        install_actions = manager.get_install_actions(name)
        hints = manager.get_hints(name)
        prompts = manager.get_prompt_bundle(name)
        return {
            "name": name,
            "metadata": asdict(metadata) if metadata else None,
            "capabilities": asdict(capabilities) if capabilities else None,
            "install_actions": [asdict(action) for action in install_actions],
            "hints": asdict(hints) if hints else None,
            "prompts": asdict(prompts) if prompts else None,
            "source": manager.get_plugin_source(name),
            "has_plugin": plugin is not None,
        }

    def _dump_plugins() -> List[Dict[str, Any]]:
        library_registry.get_all_libraries()
        items: List[Dict[str, Any]] = []
        for name in manager.list_plugin_names():
            items.append(_plugin_payload(name))
        return items

    if action_norm == "list":
        return {"success": True, "action": "list", "plugins": _dump_plugins()}
    if action_norm == "reload":
        reload_result = library_registry.reload_library_plugins()
        return {
            "success": True,
            "action": "reload",
            "reload_result": reload_result,
            "plugins": _dump_plugins(),
        }
    if action_norm == "diagnose":
        if not plugin_name:
            return {
                "success": False,
                "error": "plugin_name is required for action='diagnose'",
                "action": "diagnose",
            }
        if plugin_name not in manager.list_plugin_names():
            return {
                "success": False,
                "error": f"Plugin '{plugin_name}' not found",
                "action": "diagnose",
            }
        return {
            "success": True,
            "action": "diagnose",
            "plugin": _plugin_payload(plugin_name),
        }
    return {"success": False, "error": f"Unsupported action '{action}'"}


async def _refine_recommendations_with_llm(
    ctx: Context,
    scenario: str,
    context: str,
    recommendations: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Use LLM to refine library recommendations.

    Args:
        ctx: FastMCP context with sample() method
        scenario: Original scenario text
        context: Testing context (web, api, etc.)
        recommendations: Rule-based recommendations to refine

    Returns:
        Refined recommendations or None if refinement fails
    """
    if not ctx or not hasattr(ctx, "sample"):
        return None

    if not recommendations:
        return None

    # Build refinement prompt
    lib_names = [
        r.get("library_name", r.get("name", "Unknown")) for r in recommendations[:10]
    ]

    prompt = f"""Given this test automation scenario and candidate libraries, select the BEST 3-5 libraries.

Scenario: {scenario}
Context: {context}
Candidate libraries: {", ".join(lib_names)}

Rules:
1. NEVER include both Browser and SeleniumLibrary (they conflict)
2. Prefer Browser over SeleniumLibrary for web testing (it's more modern)
3. Only include libraries actually needed for the scenario
4. Include BuiltIn and Collections if other libraries are selected

Respond with ONLY a JSON array of library names, like: ["Browser", "BuiltIn", "Collections"]
"""

    try:
        result = await ctx.sample(
            messages=prompt,
            system_prompt="You are a Robot Framework expert. Respond only with a JSON array of library names.",
            temperature=0.2,
            max_tokens=256,
        )

        # Parse response
        import json
        import re

        response_text = result.text if hasattr(result, "text") else str(result)

        # Extract JSON array from response
        json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
        if json_match:
            selected_names = json.loads(json_match.group())

            # Filter recommendations to only include selected libraries
            refined = [
                r
                for r in recommendations
                if r.get("library_name", r.get("name")) in selected_names
            ]

            # Ensure we have at least some results
            if refined:
                logger.info(
                    f"LLM refined {len(recommendations)} recommendations to {len(refined)}"
                )
                return refined

    except Exception as e:
        logger.debug(f"LLM refinement parsing failed: {e}")

    return None


@mcp.tool
async def recommend_libraries(
    scenario: str,
    context: str = "web",
    session_id: str | None = None,
    max_recommendations: int = 5,
    check_availability: bool = True,
    apply_search_order: bool = True,
    mode: str = "direct",
    samples: List[Dict[str, Any]] | None = None,
    k: int | None = None,
    available_libraries: List[Dict[str, Any]] | None = None,
    include_keywords: bool = True,
    use_llm_refinement: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Recommend libraries for a scenario or generate/merge sampling prompts.

    WHEN TO USE THIS TOOL:
    - IMMEDIATELY after analyze_scenario, before execute_step
    - When you encounter "No keyword with name" errors
    - To discover which libraries provide needed functionality

    This tool analyzes scenario text and suggests relevant libraries, saving you from
    guessing which libraries to import.

    Args:
        scenario: Natural-language description of the task to automate.
        context: Context such as "web", "mobile", or "api". Defaults to "web".
        session_id: Optional session id to align recommendations with an existing session.
        max_recommendations: Maximum libraries to return (direct mode).
        check_availability: When True, checks installability/presence of suggested libs.
        apply_search_order: When True, applies recommended order to the session.
        mode: "direct", "sampling_prompt", or "merge_samples".
        samples: Sampled recommendations to merge when mode="merge_samples".
        k: Number of samples to request when mode="sampling_prompt" (defaults to 4).
        available_libraries: Optional pre-fetched library metadata to use instead of registry defaults.
        include_keywords: When True, include a compact keyword list (names only) for the top recommendation.
        use_llm_refinement: When True, uses LLM via ctx.sample() to refine recommendations.
        ctx: FastMCP Context object providing access to sample() for LLM refinement.

    Returns:
        Dict[str, Any]: Recommendation payload:
            - success: bool
            - recommendations or sampling_prompt or merged result (depending on mode)
            - session_id: echoed/preserved when provided
            - error/guidance: present on failure

    Examples:
        After analyzing web scenario:
            recommend_libraries(
                scenario="Test login form with username and password",
                context="web",
                session_id="web_test"
            )
            # Returns: ["Browser", "SeleniumLibrary"] with usage guidance
    """

    mode_norm = (mode or "direct").strip().lower()
    if mode_norm in {"sampling", "sampling_prompt"}:
        from robotmcp.config.library_registry import get_recommendation_info

        libs = available_libraries or get_recommendation_info()
        for lib in libs:
            lib.setdefault("conflicts", [])
        sample_count = k or 4
        prompt_text = _build_recommend_libraries_sampling_prompt(
            scenario, sample_count, libs
        )
        return {
            "success": True,
            "mode": "sampling_prompt",
            "prompt": prompt_text,
            "available_libraries": libs,
            "recommended_sampling": {"count": sample_count, "temperature": 0.4},
        }

    if mode_norm in {"merge", "merge_samples"}:
        if not samples:
            return {
                "success": False,
                "mode": "merge_samples",
                "error": "samples are required when mode='merge_samples'",
            }
        prompt_text = _build_choose_recommendations_prompt(samples)
        return {
            "success": True,
            "mode": "merge_samples",
            "prompt": prompt_text,
        }

    # Get explicit library preference from session if available
    explicit_pref = None
    if session_id:
        try:
            session_for_pref = execution_engine.session_manager.get_session(session_id)
            if session_for_pref and hasattr(
                session_for_pref, "explicit_library_preference"
            ):
                explicit_pref = session_for_pref.explicit_library_preference
        except Exception:
            pass

    rec = library_recommender.recommend_libraries(
        scenario,
        context=context,
        max_recommendations=max_recommendations,
        explicit_library_preference=explicit_pref,
    )
    if not rec.get("success"):
        return {"success": False, "error": rec.get("error", "Recommendation failed")}

    recommendations = rec.get("recommendations", [])
    recommended_names = [
        r.get("library_name") for r in recommendations if r.get("library_name")
    ]

    def _attach_keywords(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not recs:
            return recs
        # Only enrich the top recommendation to keep payload small
        target = recs[0]
        lib_name = target.get("library_name")
        if not lib_name:
            return recs

        from robotmcp.plugins.manager import get_library_plugin_manager
        from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

        keywords: List[str] = []
        source = "none"

        # Try plugin hints first
        mgr = get_library_plugin_manager()
        hints = mgr.get_hints(lib_name)
        if hints and hints.standard_keywords:
            keywords = list(hints.standard_keywords)
            source = "plugin_hints"
        else:
            # Fallback to libdoc cache (full keyword list to avoid truncation)
            try:
                storage = get_rf_doc_storage()
                kw_docs = storage.get_keywords_by_library(lib_name) or []
                keywords = [kw.name for kw in kw_docs]
                if keywords:
                    source = "libdoc_cache"
            except Exception:
                keywords = []
                source = "none"

        if keywords:
            target["keywords"] = keywords
            target["keyword_source"] = source
            target["keyword_hint"] = (
                "Call get_keyword_info for keyword arguments and documentation."
            )
        return recs

    result: Dict[str, Any] = {
        "success": True,
        "mode": "direct",
        "scenario": scenario,
        "context": context,
        "recommended_libraries": recommended_names,
        "recommendations": _attach_keywords(recommendations)
        if include_keywords
        else recommendations,
    }

    availability_info = None
    if check_availability and recommended_names:
        availability_info = execution_engine.check_library_requirements(
            recommended_names
        )
        result["availability"] = availability_info

    session = None
    if session_id:
        session = execution_engine.session_manager.get_or_create_session(session_id)

    auto_imported: List[str] = []
    auto_import_errors: List[Dict[str, Any]] = []

    if session and recommended_names:
        explicit = getattr(session, "explicit_library_preference", None)

        if explicit:
            recommended_names = [explicit] + [
                n for n in recommended_names if n != explicit
            ]
            if explicit == "SeleniumLibrary" and "Browser" in recommended_names:
                recommended_names = [n for n in recommended_names if n != "Browser"]
            if explicit == "Browser" and "SeleniumLibrary" in recommended_names:
                recommended_names = [
                    n for n in recommended_names if n != "SeleniumLibrary"
                ]

            name_to_rec = {r.get("library_name"): r for r in recommendations}
            recommendations = [
                name_to_rec[n] for n in recommended_names if n in name_to_rec
            ]
            result["recommendations"] = recommendations
            result["recommended_libraries"] = recommended_names

        for lib in recommended_names:
            try:
                session.import_library(lib, force=True)
            except Exception as e:
                logger.debug(f"Could not import {lib} into session {session_id}: {e}")

        available_set = set(
            (availability_info.get("available_libraries") or [])
            if availability_info
            else []
        )

        rf_mgr = get_rf_native_context_manager()
        processed: set[str] = set()
        for entry in recommendations:
            name = entry.get("library_name")
            if not name or name in processed:
                continue
            processed.add(name)
            if entry.get("is_builtin"):
                continue
            if availability_info and name not in available_set:
                continue
            try:
                import_result = rf_mgr.import_library_for_session(
                    session_id, name, args=(), alias=None
                )
            except Exception as exc:  # pragma: no cover - defensive
                auto_import_errors.append({"library": name, "error": str(exc)})
                continue
            if import_result.get("success"):
                auto_imported.append(name)
            else:
                auto_import_errors.append(
                    {"library": name, "error": import_result.get("error")}
                )

        session_setup_info: Dict[str, Any] = {
            "session_id": session_id,
            "auto_imports": {
                "imported": auto_imported,
                "errors": auto_import_errors,
            },
        }

        if apply_search_order:
            old_order = session.get_search_order()
            preferred = (
                availability_info.get("available_libraries", [])
                if availability_info
                else recommended_names
            )
            if explicit:
                preferred = [explicit] + [n for n in preferred if n != explicit]
                if explicit == "SeleniumLibrary":
                    preferred = [n for n in preferred if n != "Browser"]
                if explicit == "Browser":
                    preferred = [n for n in preferred if n != "SeleniumLibrary"]
            new_order = list(
                dict.fromkeys(
                    preferred + [lib for lib in old_order if lib not in preferred]
                )
            )
            session.set_library_search_order(new_order)
            session_setup_info.update(
                {
                    "old_search_order": old_order,
                    "new_search_order": new_order,
                    "applied": True,
                }
            )
        else:
            session_setup_info["applied"] = False

        result["session_setup"] = session_setup_info

    # Use LLM to refine/generate recommendations when enabled
    from robotmcp.utils.sampling import is_sampling_enabled as _sampling_on

    _do_llm = use_llm_refinement or (_sampling_on() and ctx is not None)
    if _do_llm:
        try:
            from robotmcp.utils.sampling import sample_recommend_libraries

            llm_recs = await sample_recommend_libraries(
                ctx, scenario, context, result.get("recommendations", [])
            )
            if llm_recs:
                # Use LLM recommendations, keeping rule-based metadata
                rule_recs = {
                    r.get("library_name", r.get("name", "")): r
                    for r in result.get("recommendations", [])
                }
                merged = []
                for llm_rec in llm_recs:
                    lib_name = llm_rec.get("library_name", "")
                    if lib_name in rule_recs:
                        # Enrich rule-based rec with LLM confidence/rationale
                        enriched = dict(rule_recs[lib_name])
                        enriched["llm_confidence"] = llm_rec.get("confidence", 0.8)
                        enriched["llm_rationale"] = llm_rec.get("rationale", "")
                        merged.append(enriched)
                    else:
                        merged.append(llm_rec)
                if merged:
                    result["recommendations"] = merged
                    result["recommended_libraries"] = [
                        r.get("library_name", r.get("name", "")) for r in merged
                    ]
                    result["llm_refined"] = True
                    result["sampling_enhanced"] = True
                    logger.info(f"LLM sampling generated {len(merged)} recommendations")
                else:
                    # Fallback to existing refinement
                    refined = await _refine_recommendations_with_llm(
                        ctx, scenario, context, result.get("recommendations", [])
                    )
                    if refined:
                        result["recommendations"] = refined
                        result["llm_refined"] = True
            else:
                # Fallback to existing refinement
                refined = await _refine_recommendations_with_llm(
                    ctx, scenario, context, result.get("recommendations", [])
                )
                if refined:
                    result["recommendations"] = refined
                    result["llm_refined"] = True
        except Exception as e:
            logger.debug(f"LLM recommendation failed, using rule-based: {e}")
            result["llm_refined"] = False

    # Track for instruction learning
    if session_id:
        _track_tool_result(
            session_id,
            "recommend_libraries",
            {"scenario": scenario, "context": context},
            result,
        )

    return result


@mcp.tool
async def analyze_scenario(
    scenario: str, context: str = "web", session_id: str = None, ctx: Context = None
) -> Dict[str, Any]:
    """Analyze a natural-language scenario into structured intent and create a session.

    WORKFLOW: This should be your FIRST tool call for any test scenario.

    What this tool does:
    1. Creates a new session with unique session_id (or reuses provided one)
    2. Analyzes scenario to detect context (web/api/mobile/desktop)
    3. Auto-configures libraries based on scenario text
    4. Returns session_id for use in ALL subsequent tool calls

    CRITICAL: Save the session_id from the response and use it in all other tool calls.

    Args:
        scenario: Human-language description of the task to automate.
        context: Application context (e.g., "web", "mobile", "api"); defaults to "web".
        session_id: Optional existing session id to reuse; if omitted, a new one is created.

    Returns:
        Dict[str, Any]: Structured intent and session metadata:
            - success: bool
            - session_id: created/resolved id (reuse for subsequent tools)
            - session_info: auto-configured libraries, search order, next-step guidance
            - intent/requirements/risk: parsed scenario details
            - error/guidance: present on failure

    Examples:
        analyze_scenario(
            scenario="Test REST API endpoint /users with GET request",
            context="api"
        )
        # Returns: {"session_id": "auto_generated_id", "recommended_libraries": ["RequestsLibrary"], ...}

        # Use session_id in ALL subsequent calls:
        execute_step(
            keyword="Create Session",
            arguments=["api", "https://api.example.com"],
            session_id="auto_generated_id"  # Use the returned session_id
        )
    """
    # Analyze the scenario first
    result = await nlp_processor.analyze_scenario(scenario, context)

    # Enhance with LLM sampling when feature flag is enabled
    from robotmcp.utils.sampling import is_sampling_enabled

    if is_sampling_enabled() and ctx:
        try:
            from robotmcp.utils.sampling import sample_analyze_scenario

            sampling_result = await sample_analyze_scenario(ctx, scenario, context)
            if sampling_result:
                # Merge LLM insights into rule-based result
                result["sampling_enhanced"] = True
                analysis = result.get("analysis", {})
                if sampling_result.get("session_type"):
                    analysis["detected_session_type"] = sampling_result["session_type"]
                if sampling_result.get("primary_library"):
                    analysis["explicit_library_preference"] = sampling_result[
                        "primary_library"
                    ]
                if sampling_result.get("library_preference"):
                    analysis["explicit_library_preference"] = sampling_result[
                        "library_preference"
                    ]
                if sampling_result.get("detected_context"):
                    analysis["detected_context"] = sampling_result["detected_context"]
                result["analysis"] = analysis
                logger.info("Scenario analysis enhanced with LLM sampling")
        except Exception as e:
            logger.debug(f"LLM sampling enhancement failed, using rule-based: {e}")
            result["sampling_enhanced"] = False

    # ALWAYS create a session - either use provided ID or generate one
    if not session_id:
        session_id = execution_engine.session_manager.create_session_id()
        logger.info(f"Auto-generated session ID: {session_id}")
    else:
        logger.info(f"Using provided session ID: {session_id}")

    logger.info(
        f"Creating and auto-configuring session '{session_id}' based on scenario analysis"
    )

    # Get or create session using execution coordinator
    session = execution_engine.session_manager.get_or_create_session(session_id)

    # --- Attach bridge awareness (R2) ---
    # When a Debug Attach Bridge is configured, pre-populate the session from
    # the live RF process instead of relying solely on NLP scenario analysis.
    # This prevents accidental analyze_scenario calls from creating sessions
    # that are disconnected from the actual RF state.
    attach_bridge_active = False
    attach_client = _get_external_client_if_configured()
    if attach_client is not None:
        try:
            diag = attach_client.diagnostics()
            if diag.get("success"):
                attach_bridge_active = True
                bridge_libs = diag.get("result", {}).get("libraries", [])
                logger.info(
                    f"Attach bridge active: pre-populating session '{session_id}' "
                    f"with {len(bridge_libs)} libraries from live RF process"
                )
                # Import libraries discovered from the live RF process
                for lib_name in bridge_libs:
                    if lib_name not in session.imported_libraries:
                        try:
                            session.import_library(lib_name)
                            session.loaded_libraries.add(lib_name)
                        except Exception:
                            pass
                # Sync variables from the live RF process
                try:
                    var_resp = attach_client.get_variables()
                    if var_resp.get("success"):
                        bridge_vars = var_resp.get("result", {})
                        for var_name, var_value in bridge_vars.items():
                            # Strip ${} wrapper for session storage
                            base = var_name
                            if base.startswith("${") and base.endswith("}"):
                                base = base[2:-1]
                            session.variables[base] = var_value
                except Exception as ve:
                    logger.debug(f"Could not sync variables from bridge: {ve}")
                # Mark libraries as loaded since they come from the live process
                session.libraries_loaded = True
        except Exception as be:
            logger.debug(f"Attach bridge diagnostics failed: {be}")

    # Detect platform type from scenario
    platform_type = execution_engine.session_manager.detect_platform_from_scenario(
        scenario
    )

    # Initialize mobile session if detected
    if platform_type == PlatformType.MOBILE:
        execution_engine.session_manager.initialize_mobile_session(session, scenario)
        logger.info(
            f"Initialized mobile session for platform: {session.mobile_config.platform_name if session.mobile_config else 'Unknown'}"
        )
    else:
        # Auto-configure session based on scenario (existing web flow)
        session.configure_from_scenario(scenario)

    # Override session config with LLM-detected preferences when sampling enabled
    from robotmcp.utils.sampling import is_sampling_enabled as _is_sampling_on

    if _is_sampling_on() and ctx:
        try:
            from robotmcp.utils.sampling import (
                sample_detect_library_preference,
                sample_detect_session_type,
            )

            llm_lib_pref = await sample_detect_library_preference(ctx, scenario)
            if llm_lib_pref:
                session.explicit_library_preference = llm_lib_pref
                logger.info(f"LLM sampling detected library preference: {llm_lib_pref}")
            llm_session_type = await sample_detect_session_type(ctx, scenario, context)
            if llm_session_type:
                from robotmcp.models.session_models import SessionType

                try:
                    session.session_type = SessionType(llm_session_type)
                    logger.info(
                        f"LLM sampling detected session type: {llm_session_type}"
                    )
                except ValueError:
                    pass
        except Exception as e:
            logger.debug(f"LLM session config enhancement failed: {e}")

    # Enhanced session info with guidance
    result["session_info"] = {
        "session_id": session_id,
        "auto_configured": session.auto_configured,
        "session_type": session.session_type.value,
        "explicit_library_preference": session.explicit_library_preference,
        "recommended_libraries": session.get_libraries_to_load(),
        "search_order": session.get_search_order(),
        "libraries_loaded": list(session.loaded_libraries),
        "next_step_guidance": f"Use session_id='{session_id}' in all subsequent tool calls",
        "status": "active",
        "ready_for_execution": True,
        "attach_bridge_active": attach_bridge_active,
    }
    if attach_bridge_active:
        result["session_info"]["attach_note"] = (
            "Session was pre-populated from the live RF process via Debug Attach Bridge. "
            "Libraries and variables reflect the actual RF runtime state."
        )
    result["session_info"]["recommended_tools"] = [
        {
            "order": idx,
            "tool": tool,
            "description": description,
        }
        for idx, (tool, description) in enumerate(AUTOMATION_TOOL_GUIDE, start=1)
    ]

    logger.info(
        f"Session '{session_id}' configured: type={session.session_type.value}, preference={session.explicit_library_preference}"
    )

    result["session_id"] = session_id
    result["context"] = context

    # Start instruction learning tracking for this session
    try:
        hooks = _get_instruction_hooks()
        # Detect scenario type from context
        scenario_type = "unknown"
        if context == "web":
            scenario_type = "web_automation"
        elif context == "api":
            scenario_type = "api_testing"
        elif context == "mobile":
            scenario_type = "mobile_testing"
        elif context == "desktop":
            scenario_type = "desktop_automation"

        hooks.on_session_start(
            session_id=session_id,
            instruction_mode=os.environ.get("ROBOTMCP_INSTRUCTION_MODE", "default"),
            scenario_type=scenario_type,
        )
        # Track the analyze_scenario call
        _track_tool_result(
            session_id,
            "analyze_scenario",
            {"scenario": scenario, "context": context},
            result,
        )
    except Exception as e:
        logger.debug(f"Failed to start instruction learning: {e}")

    return result


def _filter_keywords_by_session_library(
    keywords: List[Dict[str, Any]],
    session_id: str | None,
    session_library_preference: str | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Filter keywords to only include those compatible with session library preference.

    Uses plugin system to determine incompatible libraries and keyword alternatives.

    Args:
        keywords: List of keyword dicts with 'name' and 'library' fields
        session_id: Session ID for logging
        session_library_preference: Explicit library preference (e.g., "Browser", "SeleniumLibrary")

    Returns:
        Tuple of (filtered_keywords, excluded_keywords_with_alternatives)
    """
    if not session_library_preference or not keywords:
        return keywords, []

    # Get incompatible libraries from plugin
    plugin_manager = get_library_plugin_manager()
    excluded_libraries = set(
        plugin_manager.get_incompatible_libraries(session_library_preference)
    )

    if not excluded_libraries:
        return keywords, []

    # Get keyword alternatives from plugin for helpful messages
    keyword_alternatives = plugin_manager.get_keyword_alternatives(
        session_library_preference
    )

    filtered_keywords = []
    excluded_with_alternatives = []

    for kw in keywords:
        kw_library = kw.get("library", "")
        kw_name = kw.get("name", "")

        if kw_library in excluded_libraries:
            # This keyword is from an incompatible library
            kw_lower = kw_name.lower()
            alt_info = keyword_alternatives.get(kw_lower, {})

            excluded_info = {
                "keyword": kw_name,
                "incompatible_library": kw_library,
                "session_library": session_library_preference,
                "reason": f"Keyword '{kw_name}' is from {kw_library}, but session uses {session_library_preference}",
            }

            if alt_info:
                excluded_info["alternative"] = alt_info.get("alternative")
                excluded_info["example"] = alt_info.get("example")

            excluded_with_alternatives.append(excluded_info)
        else:
            filtered_keywords.append(kw)

    if excluded_with_alternatives:
        logger.info(
            f"Session {session_id}: Filtered out {len(excluded_with_alternatives)} "
            f"keywords from incompatible libraries: {list(excluded_libraries)}"
        )

    return filtered_keywords, excluded_with_alternatives


@mcp.tool
async def find_keywords(
    query: str,
    strategy: str = "semantic",
    context: str = "web",
    session_id: str | None = None,
    library_name: str | None = None,
    current_state: Dict[str, Any] | None = None,
    limit: int | None = None,
) -> Dict[str, Any]:
    """Discover Robot Framework keywords using multiple strategies.

    WHEN TO USE THIS TOOL:
    - ALWAYS before calling execute_step with an unfamiliar keyword
    - When you're unsure of exact keyword name or spelling
    - To discover what keywords are available in imported libraries
    - When error says "No keyword with name 'X' found"

    Args:
        query: Search text or intent description.
               Examples: "click a button", "validate json", "get*request"
        strategy: Discovery approach:
                  - "semantic": Natural language search (best for exploring)
                  - "pattern": Glob/regex matching (best when you know partial name)
                  - "catalog": List all available keywords
                  - "session": List keywords from session's loaded libraries
        context: Scenario context (e.g., "web", "mobile", "api") used by semantic discovery.
        session_id: Required for strategy="session" to search the live RF namespace.
        library_name: Optional library filter for catalog search.
        current_state: Optional state payload to improve semantic matching.
        limit: Optional maximum number of results to return.

    Returns:
        Dict[str, Any]: Discovery result:
            - success: bool
            - strategy: strategy used
            - query: original query
            - result/results: strategy-specific payload
            - error: present on failure

    Examples:
        Discover button-clicking keywords:
            find_keywords(
                query="click a button",
                strategy="semantic",
                session_id="web_session"
            )
            # Returns: ["Click", "Click Button", "Click Element", ...]

        Find keywords matching pattern:
            find_keywords(
                query="Get*",
                strategy="pattern",
                session_id="api_session"
            )
            # Returns: ["GET", "Get Request", "Get Element", ...]
    """

    strategy_norm = (strategy or "semantic").strip().lower()
    current_state = current_state or {}
    limit_value: int | None = None
    if limit is not None:
        try:
            limit_value = int(limit)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            limit_value = None

    # Get session library preference for filtering
    session_library_preference = None
    if session_id:
        session = execution_engine.session_manager.get_session(session_id)
        if session:
            session_library_preference = getattr(
                session, "explicit_library_preference", None
            )

    if strategy_norm in {"semantic", "intent"}:
        discovery = await keyword_matcher.discover_keywords(
            query, context, current_state
        )

        # Apply library filtering if session has preference
        excluded = []
        if session_library_preference and discovery.get("matches"):
            # Convert semantic matches to format expected by filter function
            matches_for_filter = [
                {"name": m.get("keyword_name"), "library": m.get("library"), **m}
                for m in discovery.get("matches", [])
            ]
            filtered_matches, excluded = _filter_keywords_by_session_library(
                matches_for_filter, session_id, session_library_preference
            )
            # Convert back to original format
            discovery["matches"] = [
                {k: v for k, v in m.items() if k not in ("name",)}
                for m in filtered_matches
            ]
            discovery["filtered_count"] = len(excluded)

        result = {
            "success": bool(discovery.get("success", True)),
            "strategy": "semantic",
            "query": query,
            "result": discovery,
        }
        if excluded:
            result["excluded_keywords"] = excluded
            result["session_library"] = session_library_preference
        # Track for instruction learning
        if session_id:
            _track_tool_result(
                session_id,
                "find_keywords",
                {"query": query, "strategy": strategy},
                result,
            )
        return result

    if strategy_norm in {"pattern", "search"}:
        await _ensure_all_session_libraries_loaded()
        matches = execution_engine.search_keywords(query)

        # Apply library filtering if session has preference
        excluded = []
        if session_library_preference:
            matches, excluded = _filter_keywords_by_session_library(
                matches, session_id, session_library_preference
            )

        if limit_value is not None:
            matches = matches[:limit_value]

        result = {
            "success": True,
            "strategy": "pattern",
            "query": query,
            "results": matches,
        }
        if excluded:
            result["excluded_keywords"] = excluded
            result["session_library"] = session_library_preference
        # Track for instruction learning
        if session_id:
            _track_tool_result(
                session_id,
                "find_keywords",
                {"query": query, "strategy": strategy},
                result,
            )
        return result

    if strategy_norm in {"catalog", "library"}:
        await _ensure_all_session_libraries_loaded()
        catalog = execution_engine.get_available_keywords(library_name)
        if query:
            lowered = query.lower()
            catalog = [
                item
                for item in catalog
                if lowered in (item.get("name") or "").lower()
                or lowered in (item.get("library") or "").lower()
            ]

        # Apply library filtering if session has preference
        excluded = []
        if session_library_preference:
            catalog, excluded = _filter_keywords_by_session_library(
                catalog, session_id, session_library_preference
            )

        if limit_value is not None:
            catalog = catalog[:limit_value]

        result = {
            "success": True,
            "strategy": "catalog",
            "query": query,
            "library": library_name,
            "results": catalog,
        }
        if excluded:
            result["excluded_keywords"] = excluded
            result["session_library"] = session_library_preference
        # Track for instruction learning
        if session_id:
            _track_tool_result(
                session_id,
                "find_keywords",
                {"query": query, "strategy": strategy},
                result,
            )
        return result

    if strategy_norm in {"session", "namespace"}:
        if not session_id:
            return {
                "success": False,
                "error": "session_id is required when strategy='session'",
            }
        mgr = get_rf_native_context_manager()
        payload = mgr.list_available_keywords(session_id)
        payload.update({"strategy": "session", "query": query})
        # Track for instruction learning
        _track_tool_result(
            session_id, "find_keywords", {"query": query, "strategy": strategy}, payload
        )
        return payload

    return {"success": False, "error": f"Unsupported strategy '{strategy}'"}


@mcp.tool(enabled=False)
async def discover_keywords(
    action_description: str, context: str = "web", current_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Find matching Robot Framework keywords for an action.

    Args:
        action_description: Description of the action to perform
        context: Current context (web, mobile, API, etc.)
        current_state: Current application state
    """
    if current_state is None:
        current_state = {}
    return await keyword_matcher.discover_keywords(
        action_description, context, current_state
    )


@mcp.tool
async def manage_session(
    action: str,
    session_id: str,
    libraries: List[str] | None = None,
    variables: Dict[str, Any] | List[str] | None = None,
    resource_path: str | None = None,
    library_name: str | None = None,
    args: List[str] | None = None,
    alias: str | None = None,
    scope: Literal["test", "suite", "global"] = "suite",
    variable_file_path: str | None = None,
    # ADR-005: Multi-test parameters
    test_name: str | None = None,
    test_documentation: str = "",
    test_tags: List[str] | None = None,
    test_setup: Dict[str, Any] | None = None,
    test_teardown: Dict[str, Any] | None = None,
    test_status: str = "pass",
    test_message: str = "",
    keyword: str | None = None,
) -> Dict[str, Any]:
    """Manage session lifecycle: initialize, configure libraries/variables, and organize tests.

    Workflows:
        Single test:  init -> execute_step (repeat) -> build_test_suite
        Multi-test:   init -> set_suite_setup -> start_test -> execute_step (repeat)
                      -> end_test -> start_test -> ... -> build_test_suite

    Actions and parameters (session_id is always required):

        init             - Create session and load libraries.
                           Params: libraries (list of library names),
                                   variables (dict or list to pre-set).

        import_library   - Add a library to an existing session.
                           Params: library_name, args (constructor args), alias.

        import_resource  - Import a Robot Framework resource file.
                           Params: resource_path, args.

        set_variables    - Set variables in the session.
                           Params: variables (dict {"NAME": "value"} or list ["NAME=value"]),
                                   scope ("test" | "suite" | "global", default "suite").

        import_variables - Load variables from a Python variable file.
                           Params: variable_file_path, args (passed to get_variables()).

        start_test       - Begin a named test (enables multi-test mode). Local mode only.
                           Params: test_name (required),
                                   test_documentation, test_tags,
                                   test_setup (dict {"keyword": "...", "arguments": [...]}),
                                   test_teardown (same format as test_setup).
                           Alias: start_task.

        end_test         - End the current test. Local mode only.
                           Params: test_status ("pass" or "fail", default "pass"),
                                   test_message (optional error description).
                           NOTE: test_status and test_message are session tracking metadata.
                           They do NOT affect the .robot file generated by build_test_suite.
                           Alias: end_task.

        list_tests       - List all tests in the session with their status and step counts.
                           Params: (none).

        set_suite_setup    - Set a suite-level setup keyword (appears in *** Settings ***).
                             Params: keyword (required), args (keyword arguments).

        set_suite_teardown - Set a suite-level teardown keyword (appears in *** Settings ***).
                             Params: keyword (required), args (keyword arguments).

    Returns:
        Dict with success, session_id, and action-specific details.
        On failure: error and guidance fields are present.

    Examples:
        Initialize session with libraries:
            manage_session(action="init", session_id="s1",
                           libraries=["Browser", "BuiltIn", "Collections"])

        Set suite-level variables:
            manage_session(action="set_variables", session_id="s1",
                           variables={"BASE_URL": "https://example.com", "TIMEOUT": "30"})

        Import a library with constructor arguments:
            manage_session(action="import_library", session_id="s1",
                           library_name="Browser", args=["chromium"])

        Load a Python variable file:
            manage_session(action="import_variables", session_id="s1",
                           variable_file_path="config/variables.py",
                           args=["production", "secret_key"])

        Start a named test (multi-test mode):
            manage_session(action="start_test", session_id="s1",
                           test_name="Login Test", test_tags=["smoke"],
                           test_setup={"keyword": "Open Browser", "arguments": ["chromium"]})

        End the current test:
            manage_session(action="end_test", session_id="s1")

        Set suite setup (for generated .robot file):
            manage_session(action="set_suite_setup", session_id="s1",
                           keyword="New Browser", args=["chromium"])

        Set suite teardown:
            manage_session(action="set_suite_teardown", session_id="s1",
                           keyword="Close Browser")
    """

    action_norm = (action or "").strip().lower()
    session = execution_engine.session_manager.get_or_create_session(session_id)

    if action_norm in {"init", "initialize", "bootstrap"}:
        # Start instruction learning tracking for this session
        try:
            hooks = _get_instruction_hooks()
            # Detect scenario type from libraries or use default
            scenario_type = "unknown"
            if libraries:
                if any(
                    "browser" in lib.lower() or "selenium" in lib.lower()
                    for lib in libraries
                ):
                    scenario_type = "web_automation"
                elif any(
                    "api" in lib.lower() or "request" in lib.lower()
                    for lib in libraries
                ):
                    scenario_type = "api_testing"
                elif any(
                    "mobile" in lib.lower() or "appium" in lib.lower()
                    for lib in libraries
                ):
                    scenario_type = "mobile_testing"

            hooks.on_session_start(
                session_id=session_id,
                instruction_mode=os.environ.get("ROBOTMCP_INSTRUCTION_MODE", "default"),
                scenario_type=scenario_type,
            )
        except Exception as e:
            logger.debug(f"Failed to start instruction learning: {e}")

        loaded: List[str] = []
        problems: List[Dict[str, Any]] = []

        if libraries:
            # Ensure BuiltIn is always included (standard RF behaviour)
            if "BuiltIn" not in libraries:
                libraries = ["BuiltIn"] + list(libraries)
            for library in libraries:
                try:
                    session.import_library(library)
                    session.loaded_libraries.add(library)
                    loaded.append(library)
                except Exception as lib_error:
                    problems.append({"library": library, "error": str(lib_error)})
        else:
            # Even when no libraries are specified, ensure BuiltIn is present
            try:
                session.import_library("BuiltIn")
                session.loaded_libraries.add("BuiltIn")
                loaded.append("BuiltIn")
            except Exception as lib_error:
                problems.append({"library": "BuiltIn", "error": str(lib_error)})

        set_vars: List[str] = []
        if variables:
            if isinstance(variables, dict):
                iterable = variables.items()
            else:
                iterable = []
                for item in variables:
                    if isinstance(item, str) and "=" in item:
                        name, value = item.split("=", 1)
                        iterable.append((name, value))
            # Ensure suite_level_variables set exists for variable tracking
            if (
                not hasattr(session, "suite_level_variables")
                or session.suite_level_variables is None
            ):
                session.suite_level_variables = set()
            for name, value in iterable:
                key = name if name.startswith("${") else f"${{{name}}}"
                session.set_variable(key, value)
                set_vars.append(name)
                # Track for *** Variables *** section in generated test suite
                session.suite_level_variables.add(name)

        result = {
            "success": True,
            "action": "init",
            "session_id": session_id,
            "libraries_loaded": list(session.loaded_libraries),
            "variables_set": set_vars,
            "import_issues": problems,
            "note": "Context mode is managed via session namespace; use execute_step(use_context=True) when needed.",
        }
        _track_tool_result(
            session_id,
            "manage_session",
            {"action": action, "libraries": libraries},
            result,
        )
        return result

    if action_norm in {"import_resource", "resource"}:
        if not resource_path:
            return {"success": False, "error": "resource_path is required"}

        def _store_session_variables(
            session, variables_map: Dict[str, Any]
        ) -> List[str]:
            loaded: List[str] = []
            for name, value in variables_map.items():
                base = (
                    name[2:-1] if name.startswith("${") and name.endswith("}") else name
                )
                decorated = f"${{{base}}}"
                session.variables[base] = value
                session.variables[decorated] = value
                loaded.append(base)
            return loaded

        def _local_call() -> Dict[str, Any]:
            mgr = get_rf_native_context_manager()
            return mgr.import_resource_for_session(session_id, resource_path)

        def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
            return client.import_resource(resource_path)

        result = _call_attach_tool_with_fallback(
            "import_resource", _external_call, _local_call
        )
        # Sync returned variables (local mode only) into the ExecutionSession store
        try:
            session = execution_engine.session_manager.get_or_create_session(session_id)
            variables_map = result.get("variables_map") or {}
            if variables_map:
                _store_session_variables(session, variables_map)
            for vf in result.get("variable_files", []) or []:
                key = (vf.get("path"), tuple(vf.get("args") or ()))
                existing = {
                    (item.get("path"), tuple(item.get("args") or ()))
                    for item in session.loaded_variable_files
                }
                if key not in existing:
                    session.loaded_variable_files.append(vf)
        except Exception:
            # Best-effort sync; do not fail the import if syncing fails
            pass

        result.update({"action": "import_resource", "session_id": session_id})
        return result

    if action_norm in {"import_library", "library"}:
        if not library_name:
            return {"success": False, "error": "library_name is required"}

        def _local_call() -> Dict[str, Any]:
            mgr = get_rf_native_context_manager()
            return mgr.import_library_for_session(
                session_id, library_name, tuple(args or ()), alias
            )

        def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
            return client.import_library(library_name, list(args or ()), alias)

        result = _call_attach_tool_with_fallback(
            "import_custom_library", _external_call, _local_call
        )
        result.update({"action": "import_library", "session_id": session_id})
        return result

    if action_norm in {"set_variables", "variables"}:
        data: Dict[str, Any] = {}
        if isinstance(variables, dict):
            data = variables
        elif isinstance(variables, list):
            for item in variables:
                if isinstance(item, str) and "=" in item:
                    name, value = item.split("=", 1)
                    data[name.strip()] = value

        if not data:
            return {
                "success": False,
                "action": "set_variables",
                "session_id": session_id,
                "error": "No variables to set. Provide a non-empty 'variables' parameter.",
                "hint": (
                    "Pass variables as a dict: variables={\"MY_VAR\": \"value\"} "
                    "or as a list: variables=[\"MY_VAR=value\", \"OTHER=123\"]"
                ),
            }

        set_kw = {
            "test": "Set Test Variable",
            "suite": "Set Suite Variable",
            "global": "Set Global Variable",
        }.get(scope.lower(), "Set Suite Variable")

        results: Dict[str, bool] = {}
        errors: Dict[str, str] = {}
        client = _get_external_client_if_configured()
        if client is not None:
            for name, value in data.items():
                try:
                    resp = client.set_variable(name, value, scope=scope)
                    results[name] = bool(resp.get("success"))
                    if not results[name]:
                        errors[name] = resp.get("error", "Bridge returned success=false")
                except Exception as e:
                    results[name] = False
                    errors[name] = str(e)
            result_payload: Dict[str, Any] = {
                "success": all(results.values()),
                "action": "set_variables",
                "session_id": session_id,
                "set": [k for k, v in results.items() if v],
                "scope": scope,
                "external": True,
            }
            if errors:
                result_payload["errors"] = errors
                result_payload["hint"] = (
                    "Some variables failed to set via bridge. "
                    "Check that the RF process is running and the bridge is reachable. "
                    "Use manage_attach(action='status') to verify bridge health."
                )
            return result_payload

        for name, value in data.items():
            # Pass value directly with ${name} syntax - RF 7 handles Python
            # list/dict types automatically and stores them appropriately
            res = await execution_engine.execute_step(
                set_kw,
                [f"${{{name}}}", value],
                session_id,
                detail_level="minimal",
                use_context=True,
            )
            results[name] = bool(res.get("success"))
            if not results[name]:
                errors[name] = res.get("error", "Keyword execution returned success=false")

        # Track ALL manage_session variables for *** Variables *** section generation
        # Variables set via manage_session (any scope) should be included in the Variables
        # section to ensure generated test suites are complete and executable.
        # Without this, tests would reference variables that aren't defined.
        try:
            session = execution_engine.session_manager.get_or_create_session(session_id)
            if (
                not hasattr(session, "suite_level_variables")
                or session.suite_level_variables is None
            ):
                session.suite_level_variables = set()
            for name in data.keys():
                session.suite_level_variables.add(name)
            logger.debug(
                f"Tracked {len(data)} variables from manage_session for Variables section"
            )
        except Exception as track_error:
            logger.warning(f"Failed to track manage_session variables: {track_error}")

        result_payload = {
            "success": all(results.values()),
            "action": "set_variables",
            "session_id": session_id,
            "set": [k for k, v in results.items() if v],
            "scope": scope,
        }
        if errors:
            result_payload["errors"] = errors
            result_payload["hint"] = (
                "Some variables failed to set. Ensure the session was initialized "
                "with manage_session(action='init') first and that BuiltIn library is loaded."
            )
        return result_payload

    if action_norm in {"import_variables", "load_variables"}:
        if not variable_file_path:
            return {"success": False, "error": "variable_file_path is required"}

        def _local_call() -> Dict[str, Any]:
            mgr = get_rf_native_context_manager()
            return mgr.import_variables_for_session(
                session_id, variable_file_path, args or []
            )

        def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
            return client.import_variables(variable_file_path, args or [])

        result = _call_attach_tool_with_fallback(
            "import_variables", _external_call, _local_call
        )

        # Sync returned variables (local mode only) into the ExecutionSession store
        try:
            session = execution_engine.session_manager.get_or_create_session(session_id)
            variables_map = result.get("variables_map") or {}
            if variables_map:
                for name, value in variables_map.items():
                    # Store variables in session with proper normalization
                    base = (
                        name[2:-1]
                        if name.startswith("${") and name.endswith("}")
                        else name
                    )
                    decorated = f"${{{base}}}"
                    session.variables[base] = value
                    session.variables[decorated] = value

            # Track variable file metadata in session
            variable_file_record = {
                "path": variable_file_path,
                "args": args or [],
                "variables_loaded": list(variables_map.keys()) if variables_map else [],
            }

            if not hasattr(session, "loaded_variable_files"):
                session.loaded_variable_files = []

            # Check if already imported (by path and args combination)
            existing_key = (variable_file_path, tuple(args or []))
            existing_keys = {
                (item.get("path"), tuple(item.get("args", [])))
                for item in session.loaded_variable_files
            }

            if existing_key not in existing_keys:
                session.loaded_variable_files.append(variable_file_record)

        except Exception as sync_error:
            # Best-effort sync; do not fail the import if syncing fails
            logger.warning(
                f"Failed to sync variable file metadata to session: {sync_error}"
            )
            pass

        result.update({"action": "import_variables", "session_id": session_id})
        return result

    # ── ADR-005: Multi-test session management ─────────────────────────

    if action_norm in {"start_test", "start_task"}:
        if not test_name:
            return {"success": False, "error": "test_name is required for start_test action"}

        # Guard: multi-test not supported with active attach bridge
        if _get_external_client_if_configured() is not None:
            return {
                "success": False,
                "error": (
                    "Multi-test mode is not supported with attach bridge. "
                    "Use separate sessions or generate .robot files."
                ),
            }

        session = execution_engine.session_manager.get_or_create_session(session_id)

        session.test_registry.start_test(
            name=test_name,
            documentation=test_documentation,
            tags=test_tags or [],
            setup=test_setup,
            teardown=test_teardown,
        )

        # Start test in RF context (pushes test scope)
        ctx_result = {}
        try:
            mgr = get_rf_native_context_manager()
            ctx_result = mgr.start_test_in_context(session_id, test_name, test_documentation, test_tags or [])
        except Exception as e:
            logger.warning(f"RF context start_test failed (non-fatal): {e}")
            ctx_result = {"success": False, "error": str(e)}

        return {
            "success": True,
            "session_id": session_id,
            "action": action_norm,
            "test_name": test_name,
            "context_result": ctx_result,
        }

    if action_norm in {"end_test", "end_task"}:
        # Guard: multi-test not supported with active attach bridge
        if _get_external_client_if_configured() is not None:
            return {
                "success": False,
                "error": "Multi-test mode is not supported with attach bridge.",
            }

        session = execution_engine.session_manager.get_or_create_session(session_id)
        _status = test_status.lower() if test_status else "pass"

        test_info = session.test_registry.end_test(
            status=_status,
            message=test_message,
        )

        # End test in RF context (pops test scope, sets PREV_TEST_*)
        ctx_result = {}
        try:
            mgr = get_rf_native_context_manager()
            rf_status = "PASS" if _status == "pass" else "FAIL"
            ctx_result = mgr.end_test_in_context(session_id, rf_status, test_message)
        except Exception as e:
            logger.warning(f"RF context end_test failed (non-fatal): {e}")
            ctx_result = {"success": False, "error": str(e)}

        return {
            "success": True,
            "session_id": session_id,
            "action": action_norm,
            "test_name": test_info.name if test_info else None,
            "test_status": _status,
            "context_result": ctx_result,
        }

    if action_norm in {"list_tests"}:
        session = execution_engine.session_manager.get_or_create_session(session_id)
        tests = []
        for name, ti in session.test_registry.tests.items():
            tests.append({
                "name": name,
                "status": ti.status,
                "step_count": len(ti.steps),
                "tags": ti.tags,
                "started_at": ti.started_at.isoformat() if ti.started_at else None,
                "ended_at": ti.ended_at.isoformat() if ti.ended_at else None,
                "error_message": ti.error_message,
            })
        return {
            "success": True,
            "session_id": session_id,
            "action": "list_tests",
            "multi_test_mode": session.test_registry.is_multi_test_mode(),
            "current_test": session.test_registry.current_test_name,
            "tests": tests,
            "total_tests": len(tests),
        }

    if action_norm in {"set_suite_setup"}:
        if not keyword:
            return {"success": False, "error": "keyword is required for set_suite_setup"}
        kw_args = list(args or [])
        session = execution_engine.session_manager.get_or_create_session(session_id)
        session.suite_setup = {"keyword": keyword, "arguments": kw_args}
        return {
            "success": True,
            "session_id": session_id,
            "action": "set_suite_setup",
            "keyword": keyword,
            "arguments": kw_args,
        }

    if action_norm in {"set_suite_teardown"}:
        if not keyword:
            return {"success": False, "error": "keyword is required for set_suite_teardown"}
        kw_args = list(args or [])
        session = execution_engine.session_manager.get_or_create_session(session_id)
        session.suite_teardown = {"keyword": keyword, "arguments": kw_args}
        return {
            "success": True,
            "session_id": session_id,
            "action": "set_suite_teardown",
            "keyword": keyword,
            "arguments": kw_args,
        }

    # ── End ADR-005 ──────────────────────────────────────────────────

    return {"success": False, "error": f"Unsupported action '{action}'"}


def _chunk_string(value: str, size: int) -> List[str]:
    if size <= 0:
        size = 65536
    return [value[i : i + size] for i in range(0, len(value), size)]


@mcp.tool
async def execute_flow(
    structure: str,
    session_id: str,
    condition: str | None = None,
    then_steps: List[Dict[str, Any]] | None = None,
    else_steps: List[Dict[str, Any]] | None = None,
    items: List[Any] | None = None,
    item_var: str = "item",
    stop_on_failure: bool = True,
    max_iterations: int = 1000,
    try_steps: List[Dict[str, Any]] | None = None,
    except_patterns: List[str] | None = None,
    except_steps: List[Dict[str, Any]] | None = None,
    finally_steps: List[Dict[str, Any]] | None = None,
    rethrow: bool = False,
) -> Dict[str, Any]:
    """Execute structured flow (if/for/try) within a session.

    Args:
        structure: Flow type ("if", "for", "try").
        session_id: Session id to run the flow in.
        condition: Expression for if/conditional flows.
        then_steps: Steps for the main branch (if/loop body/try block).
        else_steps: Steps for the else branch (if).
        items: Items to iterate when structure="for".
        item_var: Variable name to bind each item in for-each loops.
        stop_on_failure: Whether to stop loop/branch execution on first failure.
        max_iterations: Maximum iterations for for-each loops.
        try_steps: Steps for the try block (when structure="try").
        except_patterns: Error patterns to match for except handling.
        except_steps: Steps for the except block.
        finally_steps: Steps for the finally block.
        rethrow: Whether to rethrow after except/finally.

    Returns:
        Dict[str, Any]: Flow execution result:
            - success: bool
            - structure: flow type executed
            - session_id: echoed id
            - per-branch results/errors
    """

    structure_norm = (structure or "").strip().lower()

    if structure_norm in {"if", "conditional"}:
        return await _execute_if_impl(
            session_id=session_id,
            condition=condition or "",
            then_steps=then_steps or [],
            else_steps=else_steps or [],
            stop_on_failure=stop_on_failure,
        )

    if structure_norm in {"for", "foreach", "for_each"}:
        return await _execute_for_each_impl(
            session_id=session_id,
            items=items or [],
            steps=then_steps or [],
            item_var=item_var,
            stop_on_failure=stop_on_failure,
            max_iterations=max_iterations,
        )

    if structure_norm in {"try", "try_except", "trycatch"}:
        return await _execute_try_except_impl(
            session_id=session_id,
            try_steps=try_steps or [],
            except_patterns=except_patterns or [],
            except_steps=except_steps or [],
            finally_steps=finally_steps or [],
            rethrow=rethrow,
        )

    return {"success": False, "error": f"Unsupported flow structure '{structure}'"}


@mcp.tool
async def get_session_state(
    session_id: str,
    sections: List[str] | None = None,
    state_type: str = "all",
    elements_of_interest: List[str] | None = None,
    page_source_filtered: bool = False,
    page_source_filtering_level: str = "standard",
    include_reduced_dom: bool = True,
    include_dom_stream: bool = False,
    dom_chunk_size: int = 65536,
) -> Dict[str, Any]:
    """Retrieve aggregated session state for debugging and visibility.

    Primary uses:
        - UI inspection: DOM tree/page source/ARIA snapshots for Browser/Selenium/Appium sessions.
        - Variable inspection: current RF variables, assigned values, and context search order.
        - Validation/health checks: validation summaries, library lists, and attach/bridge status.
        - Application insight: application_state (dom/api/database) when provided by plugins.

    Args:
        session_id: Active session identifier to inspect.
        sections: Specific data blocks to include (e.g., summary, page_source, variables, application_state).
        state_type: Type of application state to fetch when requesting application_state (dom|api|database|all).
        elements_of_interest: Targeted element identifiers passed to application state collectors.
        page_source_filtered: When True, returns sanitized/filtered DOM text instead of the full source.
        page_source_filtering_level: Filtering aggressiveness for DOM output (standard|aggressive).
        include_reduced_dom: Whether to include lightweight semantic DOM (ARIA snapshots) for quick inspection.
        include_dom_stream: Chunk large page_source payloads into page_source_stream entries for easier transport.
        dom_chunk_size: Maximum size of each DOM chunk when streaming is enabled (minimum 1024 bytes).

    Returns:
        Dict[str, Any]: Payload with:
            - success: bool indicating retrieval success.
            - session_id: resolved session id.
            - sections: list of sections included.
            - data: per-section content (e.g., variables, page_source/ARIA snapshots, validation, libraries,
              application_state).
            - error: present only on failure, with guidance if available.
    """

    sections = sections or ["summary", "page_source", "variables"]
    requested = {s.lower() for s in sections}
    payload: Dict[str, Any] = {
        "success": True,
        "session_id": session_id,
        "sections": {},
        "requested": sections,
    }

    if "summary" in requested:
        summary = await _get_session_info_payload(session_id)
        payload["sections"]["summary"] = summary

    if "application_state" in requested or "state" in requested:
        app_state = await _get_application_state_payload(
            state_type=state_type,
            elements_of_interest=elements_of_interest or [],
            session_id=session_id,
        )
        payload["sections"]["application_state"] = app_state

    if "page_source" in requested:
        page_source = await _get_page_source_payload(
            session_id=session_id,
            full_source=not page_source_filtered,
            filtered=page_source_filtered,
            filtering_level=page_source_filtering_level,
            include_reduced_dom=include_reduced_dom,
        )
        if (
            include_dom_stream
            and isinstance(page_source, dict)
            and isinstance(page_source.get("page_source"), str)
        ):
            page_source["page_source_stream"] = _chunk_string(
                page_source["page_source"], max(int(dom_chunk_size), 1024)
            )
        payload["sections"]["page_source"] = page_source

    if "variables" in requested:
        variables = await _get_context_variables_payload(session_id)
        payload["sections"]["variables"] = variables

    if "validation" in requested:
        validation = await _get_session_validation_status_payload(session_id)
        payload["sections"]["validation"] = validation

    if "libraries" in requested:
        libraries = await _get_loaded_libraries_payload()
        payload["sections"]["libraries"] = libraries

    if "rf_context" in requested or "context" in requested:
        rf_context = await _diagnose_rf_context_payload(session_id)
        payload["sections"]["rf_context"] = rf_context

    # Track for instruction learning
    _track_tool_result(session_id, "get_session_state", {"sections": sections}, payload)

    return payload


@mcp.tool
async def execute_step(
    keyword: str,
    arguments: List[str] = None,
    session_id: str = "default",
    raise_on_failure: bool = True,
    detail_level: str = "minimal",
    scenario_hint: str = None,
    assign_to: Optional[Union[str, List[str]]] = None,
    use_context: bool | None = None,
    mode: str = "keyword",
    expression: str | None = None,
    timeout_ms: int | None = None,
) -> Dict[str, Any]:
    """Execute a single Robot Framework keyword (or Evaluate) within a session.

    IMPORTANT: Do NOT invent or guess keyword names. Use find_keywords or get_keyword_info
               to discover available keywords first. Common mistakes:
               - Using "Press Button" (doesn't exist - use "Click" or "Click Button")
               - Using "Verify" (doesn't exist - use "Should Be Equal" or similar)
               - Using "Validate Json" (doesn't exist - use library-specific validation)

    Args:
        keyword: Keyword name (Library.Keyword supported).
                 Use find_keywords to discover correct keyword names before calling.
        arguments: Keyword arguments; positional and named (`name=value`) supported.
        session_id: Session to execute in; resolves default if omitted.
        raise_on_failure: If True, raise on failure; otherwise return error in payload.
        detail_level: Response verbosity: "minimal" | "standard" | "full".
        scenario_hint: Optional scenario text to auto-configure libraries on first call.
        assign_to: Variable name(s) to assign the result to (string or list).
                   CRITICAL: Use this to capture results for later steps.
                   Example: assign_to="response" captures ${response} variable
        use_context: Whether to run inside RF native context; defaults via config/attach.
        mode: "keyword" (default) or "evaluate" (runs BuiltIn.Evaluate).
        expression: Expression for mode="evaluate"; falls back to keyword/first argument.
        timeout_ms: Optional timeout in milliseconds for keyword execution.
                    If not provided, uses smart defaults based on keyword type:
                    - Element actions (Click, Fill): 5000ms
                    - Navigation (Go To, New Page): 60000ms
                    - Read operations (Get Text): 2000ms
                    - API calls (GET, POST): 30000ms
                    Set to 0 or negative to disable timeout.

    Returns:
        Dict[str, Any]: Execution result:
            - success: bool
            - result/output: keyword return value or stringified output
            - assigned_variables / session_variables: when applicable
            - error/guidance: present on failure

    Examples:
        Execute with variable assignment:
            execute_step(
                keyword="GET",
                arguments=["https://api.example.com/users"],
                assign_to="api_response",  # Stores result in ${api_response}
                session_id="api_session"
            )

        Use captured variable:
            execute_step(
                keyword="Should Be Equal",
                arguments=["${api_response.status_code}", "200"],
                session_id="api_session"
            )

        Execute with custom timeout:
            execute_step(
                keyword="Go To",
                arguments=["https://slow-website.com"],
                timeout_ms=120000,  # 2 minutes for slow page
                session_id="web_session"
            )
    """
    arguments = list(arguments or [])

    # Auto-coerce non-string arguments to strings to prevent validation errors
    # Models sometimes pass boolean True instead of "True", or integers instead of strings
    coerced_arguments = []
    for arg in arguments:
        if isinstance(arg, str):
            coerced_arguments.append(arg)
        elif arg is None:
            # Skip None arguments - they're likely optional params the model shouldn't have included
            continue
        elif isinstance(arg, bool):
            # Convert boolean to RF-style string (Python bool to string)
            coerced_arguments.append(str(arg))
        else:
            # Convert other types (int, float, etc.) to string
            coerced_arguments.append(str(arg))
    arguments = coerced_arguments

    # Handle assign_to=None explicitly passed by models
    if assign_to is None:
        assign_to = None  # Explicitly set to None (no-op, but clarifies intent)

    mode_norm = (mode or "keyword").strip().lower()
    keyword_to_run = keyword

    if mode_norm == "evaluate":
        expr = expression
        if expr is None:
            if arguments:
                expr = arguments[0]
            elif keyword:
                expr = keyword
        if not expr:
            return {
                "success": False,
                "error": "expression is required when mode='evaluate'",
                "mode": mode_norm,
            }
        keyword_to_run = "Evaluate"
        arguments = [expr]
        if use_context is None:
            use_context = True

    # Determine routing based on attach mode and default settings
    client = _get_external_client_if_configured()
    effective_use_context, mode, strict = _compute_effective_use_context(
        use_context, client, keyword_to_run
    )

    # External routing path
    if client is not None and effective_use_context:
        logger.info(
            f"ATTACH mode: routing execute_step '{keyword_to_run}' to bridge at {client.host}:{client.port}"
        )
        attach_resp = client.run_keyword(keyword_to_run, arguments, assign_to)
        if not attach_resp.get("success"):
            err = attach_resp.get("error", "attach call failed")

            # Distinguish between connectivity errors and application errors.
            # Connectivity errors (bridge unreachable) should fall back to local;
            # application errors (keyword failed on bridge) should be returned
            # directly — falling back to local would fail differently because
            # the local RF context doesn't share browser/page state with the bridge.
            is_connectivity_error = (
                "connection error" in err.lower()
                or "connection refused" in err.lower()
                or "timed out" in err.lower()
                or err == "attach call failed"
                or err == "timeout"
            )

            if is_connectivity_error:
                logger.error(f"ATTACH mode connectivity error: {err}")
                if strict or mode == "force":
                    raise Exception(
                        f"Attach bridge call failed: {err}. Is MCP Serve running and token/port correct?"
                    )
                # Only fall back to local execution for connectivity errors
                logger.warning("ATTACH unreachable; falling back to local execution")
            else:
                # Application-level error from the bridge (keyword actually ran but failed).
                # Return the error directly instead of falling back to local execution,
                # which would fail differently (e.g., "Could not find active page" because
                # browser state lives in the bridge process, not locally).
                logger.error(f"ATTACH mode keyword error: {err}")
                return {
                    "success": False,
                    "keyword": keyword_to_run,
                    "arguments": arguments,
                    "assign_to": assign_to,
                    "mode": mode_norm,
                    "error": err,
                    "source": "attach_bridge",
                }
        else:
            return {
                "success": True,
                "keyword": keyword_to_run,
                "arguments": arguments,
                "assign_to": assign_to,
                "mode": mode_norm,
                "result": attach_resp.get("result"),
                "assigned": attach_resp.get("assigned"),
            }

    # Validate keyword matches session library preference using plugin system
    session = execution_engine.session_manager.get_or_create_session(session_id)
    session_library_preference = getattr(session, "explicit_library_preference", None)

    # Infer library preference from imported_libraries when no explicit preference
    if not session_library_preference:
        _imported = getattr(session, "imported_libraries", []) or []
        if "Browser" in _imported and "SeleniumLibrary" not in _imported:
            session_library_preference = "Browser"
        elif "SeleniumLibrary" in _imported and "Browser" not in _imported:
            session_library_preference = "SeleniumLibrary"

    if session_library_preference and keyword_to_run != "Evaluate":
        try:
            # Try to find the keyword's source library
            plugin_manager = get_library_plugin_manager()
            keyword_source_library = plugin_manager.get_library_for_keyword(
                keyword_to_run.lower()
            )

            # If we found a source library, validate compatibility using plugin
            if keyword_source_library:
                validation_result = plugin_manager.validate_keyword_for_session(
                    session_library_preference,
                    session,
                    keyword_to_run,
                    keyword_source_library,
                )
                if validation_result:
                    # Plugin returned an error - return it directly
                    return validation_result
        except Exception as e:
            logger.debug(f"Could not validate keyword library via plugin: {e}")

    # Local execution path
    result = await execution_engine.execute_step(
        keyword_to_run,
        arguments,
        session_id,
        detail_level,
        scenario_hint=scenario_hint,
        assign_to=assign_to,
        use_context=bool(use_context),
        timeout_ms=timeout_ms,
    )

    # For proper MCP protocol compliance, failed steps should raise exceptions
    # This ensures AI agents see failures as red/failed instead of green/successful
    if not result.get("success", False) and raise_on_failure:
        error_msg = result.get("error", f"Step '{keyword}' failed")

        # Create detailed error message including suggestions if available
        detailed_error = f"Step execution failed: {error_msg}"
        if "suggestions" in result:
            detailed_error += f"\nSuggestions: {', '.join(result['suggestions'])}"
        # Include structured hints for better guidance
        hints = result.get("hints") or []
        if hints:
            try:
                hint_lines = []
                for h in hints:
                    title = h.get("title") or "Hint"
                    message = h.get("message") or ""
                    hint_lines.append(f"- {title}: {message}")
                if hint_lines:
                    detailed_error += "\nHints:\n" + "\n".join(hint_lines)
            except Exception:
                pass
        if "step_id" in result:
            detailed_error += f"\nStep ID: {result['step_id']}"

        raise Exception(detailed_error)

    result["mode"] = mode_norm
    result.setdefault("keyword", keyword_to_run)

    # Track for instruction learning
    _track_tool_result(
        session_id,
        "execute_step",
        {"keyword": keyword_to_run, "arguments": arguments},
        result,
    )

    return result


async def _get_application_state_payload(
    state_type: str = "all",
    elements_of_interest: List[str] | None = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    if elements_of_interest is None:
        elements_of_interest = []

    # S3: Bridge path - build application state from bridge data
    client = _get_external_client_if_configured()
    if client is not None:
        try:
            result: Dict[str, Any] = {"success": True, "source": "attach_bridge"}

            if state_type in ("dom", "all"):
                # Reuse _get_page_source_payload for DOM state via bridge
                page_data = await _get_page_source_payload(
                    session_id=session_id,
                    include_reduced_dom=True,
                )
                result["dom_state"] = {
                    "page_source": page_data.get("page_source"),
                    "aria_snapshot": page_data.get("aria_snapshot"),
                    "metadata": page_data.get("metadata"),
                    "success": page_data.get("success", False),
                }

            if state_type in ("api", "all"):
                result["api_state"] = {
                    "note": "API state inspection not available via attach bridge",
                    "success": False,
                }

            if state_type in ("database", "all"):
                result["database_state"] = {
                    "note": "Database state inspection not available via attach bridge",
                    "success": False,
                }

            # Include variables from bridge
            var_data = await _get_context_variables_payload(session_id)
            if var_data.get("success"):
                result["variables"] = var_data.get("variables", {})
                result["variable_count"] = var_data.get("variable_count", 0)

            return result
        except Exception as e:
            logger.debug(f"Application state via bridge failed: {e}")

    # Local fallback
    return await state_manager.get_state(
        state_type, elements_of_interest, session_id, execution_engine
    )


@mcp.tool(enabled=False)
async def get_application_state(
    state_type: str = "all",
    elements_of_interest: List[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Retrieve current application state.

    Args:
        state_type: Type of state to retrieve (dom, api, database, all)
        elements_of_interest: Specific elements to focus on
        session_id: Session identifier
    """
    return await _get_application_state_payload(
        state_type=state_type,
        elements_of_interest=elements_of_interest,
        session_id=session_id,
    )


@mcp.tool(enabled=False)
async def suggest_next_step(
    current_state: Dict[str, Any],
    test_objective: str,
    executed_steps: List[Dict[str, Any]] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """AI-driven suggestion for next test step.

    Args:
        current_state: Current application state
        test_objective: Overall test objective
        executed_steps: Previously executed steps
        session_id: Session identifier
    """
    if executed_steps is None:
        executed_steps = []
    return await nlp_processor.suggest_next_step(
        current_state, test_objective, executed_steps, session_id
    )


@mcp.tool
async def build_test_suite(
    test_name: str,
    session_id: str = "",
    tags: List[str] = None,
    documentation: str = "",
    remove_library_prefixes: bool = True,
) -> Dict[str, Any]:
    """Generate a Robot Framework test suite from previously executed steps.

    Args:
        test_name: Name for the generated test case.
        session_id: Session containing executed steps; auto-resolves if empty/invalid.
        tags: Optional test tags.
        documentation: Optional test case documentation.
        remove_library_prefixes: Whether to strip library prefixes from keywords.

    Returns:
        Dict[str, Any]: Suite generation result:
            - success: bool
            - session_id: resolved id
            - suite: structured suite metadata
            - rf_text: generated .robot content
            - statistics/optimization_applied: summary of generated steps
            - error/guidance: present on failure
    """
    if tags is None:
        tags = []

    # Import session resolver here to avoid circular imports
    from robotmcp.utils.session_resolution import SessionResolver

    session_resolver = SessionResolver(execution_engine.session_manager)

    # Resolve session with intelligent fallback
    resolution_result = session_resolver.resolve_session_with_fallback(session_id)

    if not resolution_result["success"]:
        # Return enhanced error with guidance
        return {
            "success": False,
            "error": "Session not ready for test suite generation",
            "error_details": resolution_result["error_guidance"],
            "guidance": [
                "Create a session and execute some steps first",
                "Use the session_id returned by analyze_scenario",
                "Check session status with get_session_validation_status",
            ],
            "validation_summary": {"passed": 0, "failed": 0},
            "recommendation": "Start with analyze_scenario() to create a properly configured session",
        }

    # Use resolved session ID
    resolved_session_id = resolution_result["session_id"]

    # Build the test suite with resolved session
    result = await test_builder.build_suite(
        resolved_session_id, test_name, tags, documentation, remove_library_prefixes
    )

    # Add session resolution info to result
    if resolution_result.get("fallback_used", False):
        result["session_resolution"] = {
            "fallback_used": True,
            "original_session_id": session_id,
            "resolved_session_id": resolved_session_id,
            "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps",
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id,
        }

    return result


@mcp.tool(enabled=False)
async def validate_scenario(
    parsed_scenario: Dict[str, Any], available_libraries: List[str] = None
) -> Dict[str, Any]:
    """Pre-execution validation of scenario feasibility.

    Args:
        parsed_scenario: Parsed scenario from analyze_scenario
        available_libraries: List of available RF libraries
    """
    if available_libraries is None:
        available_libraries = []
    return await nlp_processor.validate_scenario(parsed_scenario, available_libraries)


# Note: Removed legacy disabled recommend_libraries_ tool to avoid confusion.


async def _get_page_source_payload(
    session_id: str = "default",
    full_source: bool = False,
    filtered: bool = False,
    filtering_level: str = "standard",
    include_reduced_dom: bool = True,
) -> Dict[str, Any]:
    """Get page source with ARIA snapshot, routing through bridge if available.

    Phase 3 implementation (ADR-003): Uses dedicated bridge methods (get_page_source,
    get_aria_snapshot) for cleaner and more efficient bridge communication.

    Args:
        session_id: Session identifier (used for local fallback)
        full_source: Whether to return full unfiltered source (deprecated, use filtered=False)
        filtered: Whether to apply DOM filtering to reduce content size
        filtering_level: Filtering aggressiveness - "standard" or "aggressive"
        include_reduced_dom: Whether to include ARIA snapshot for semantic DOM representation

    Returns:
        Dict with:
            - success: bool
            - source: "attach_bridge" or "local"
            - page_source: str (HTML/XML content)
            - library: str (Browser, SeleniumLibrary, AppiumLibrary, etc.)
            - metadata: {filtered: bool, full: bool}
            - aria_snapshot: {success: bool, content: str, format: str} or
                           {skipped: True} or {success: False, error: str}
    """
    # Bridge path: use dedicated get_page_source method for cleaner communication
    client = _get_external_client_if_configured()
    if client is not None:
        try:
            # Get page source via dedicated bridge method
            ps_resp = client.get_page_source()
            if ps_resp.get("success"):
                source = ps_resp.get("result", "")
                library_type = ps_resp.get("library", "unknown")

                # Apply local filtering if requested
                is_filtered = False
                if filtered and source:
                    try:
                        from robotmcp.components.execution.page_source_service import (
                            PageSourceService,
                        )

                        source = PageSourceService.filter_page_source(
                            source, filtering_level
                        )
                        is_filtered = True
                    except Exception as filter_err:
                        logger.debug(f"Page source filtering failed: {filter_err}")

                # Get ARIA snapshot if requested
                aria_payload: Dict[str, Any] = {"skipped": True}
                if include_reduced_dom:
                    if library_type == "Browser":
                        # ARIA snapshots only available with Browser Library (Playwright)
                        try:
                            aria_resp = client.get_aria_snapshot(
                                selector="css=html",
                                format_type="yaml",
                            )
                            if aria_resp.get("success"):
                                aria_payload = {
                                    "success": True,
                                    "content": aria_resp.get("result", ""),
                                    "format": aria_resp.get("format", "yaml"),
                                    "selector": "css=html",
                                    "library": "Browser",
                                    "source": "attach_bridge",
                                }
                            else:
                                aria_payload = {
                                    "success": False,
                                    "error": aria_resp.get("error", "unknown"),
                                    "note": aria_resp.get("note"),
                                }
                        except Exception as aria_err:
                            logger.debug(f"ARIA snapshot via bridge failed: {aria_err}")
                            aria_payload = {
                                "success": False,
                                "error": "aria_snapshot_failed_via_bridge",
                            }
                    else:
                        # Non-Browser library - ARIA not available
                        aria_payload = {
                            "success": False,
                            "error": "aria_not_available",
                            "note": f"ARIA snapshots are only available with Browser Library (Playwright), not {library_type}",
                        }

                return {
                    "success": True,
                    "source": "attach_bridge",
                    "page_source": source,
                    "library": library_type,
                    "metadata": {
                        "filtered": is_filtered,
                        "full": not is_filtered,
                    },
                    "aria_snapshot": aria_payload,
                }
            else:
                # Bridge returned failure - log and fall back
                logger.warning(
                    "Bridge get_page_source failed: %s; falling back to local execution",
                    ps_resp.get("error", "unknown error"),
                )
        except Exception as e:
            logger.debug(f"Bridge page source failed: {e}, falling back to local")

    # Local path - fall back to execution engine
    return await execution_engine.get_page_source(
        session_id,
        full_source,
        filtered,
        filtering_level,
        include_reduced_dom,
    )


@mcp.tool(enabled=False)
async def get_page_source(
    session_id: str = "default",
    full_source: bool = False,
    filtered: bool = False,
    filtering_level: str = "standard",
    include_reduced_dom: bool = True,
) -> Dict[str, Any]:
    """Get page source and context for a browser session with optional DOM filtering."""
    return await _get_page_source_payload(
        session_id=session_id,
        full_source=full_source,
        filtered=filtered,
        filtering_level=filtering_level,
        include_reduced_dom=include_reduced_dom,
    )


@mcp.tool
async def check_library_availability(libraries: List[str]) -> Dict[str, Any]:
    """Verify that specified Robot Framework libraries can be imported/installed.

    Recommended as step 3 after analyze_scenario and recommend_libraries; use the recommended
    names to avoid unnecessary checks.

    Args:
        libraries: Library names to verify (preferably from recommend_libraries output).

    Returns:
        Dict[str, Any]: Availability report:
            - success: bool
            - results: per-library availability/install guidance
            - error/guidance: present on failure
    """
    result = execution_engine.check_library_requirements(libraries)
    if "success" not in result:
        result["success"] = not bool(result.get("error"))
    return result


@mcp.tool(enabled=False)
async def get_library_status(library_name: str) -> Dict[str, Any]:
    """Get detailed installation status for a specific library.

    Args:
        library_name: Name of the library to check (e.g., 'Browser', 'SeleniumLibrary')

    Returns:
        Dict with detailed status and installation information
    """
    return execution_engine.get_installation_status(library_name)


@mcp.tool(enabled=False)
async def get_available_keywords(library_name: str = None) -> List[Dict[str, Any]]:
    """List available RF keywords with minimal metadata.

    Returns one entry per keyword with fields:
    - name: keyword name
    - library: library name
    - args: list of argument names
    - arg_types: list of argument types if available (empty when unknown)
    - short_doc: short documentation summary (no full docstrings)

    If `library_name` is provided, results are filtered to that library, loading it on demand if needed.
    """
    # CRITICAL FIX: Ensure all session libraries are loaded before discovery
    await _ensure_all_session_libraries_loaded()

    return execution_engine.get_available_keywords(library_name)


@mcp.tool(enabled=False)
async def search_keywords(pattern: str) -> List[Dict[str, Any]]:
    """Search for Robot Framework keywords matching a pattern using native RF libdoc.

    Uses Robot Framework's native libdoc API for accurate search results and documentation.
    Searches through keyword names, documentation, short_doc, and tags.

    CRITICAL FIX: Now ensures all session libraries are loaded before search.

    Args:
        pattern: Search pattern to match against keyword names, documentation, or tags

    Returns:
        List of matching keywords with native RF libdoc metadata including short_doc,
        argument types, deprecation status, and enhanced tag information.
    """
    # CRITICAL FIX: Ensure all session libraries are loaded before search
    await _ensure_all_session_libraries_loaded()

    return execution_engine.search_keywords(pattern)


# =====================
# Flow/Control Tools v1
# =====================


def _normalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a step dict to expected keys."""
    return {
        "keyword": step.get("keyword", ""),
        "arguments": step.get("arguments", []) or [],
        "assign_to": step.get("assign_to"),
    }


async def _run_steps_in_context(
    session_id: str,
    steps: List[Dict[str, Any]],
    stop_on_failure: bool = True,
) -> List[Dict[str, Any]]:
    """Execute a list of steps via execute_step with use_context=True and return per-step results.

    Does not raise on failure; captures each step's success/error.
    """
    results: List[Dict[str, Any]] = []
    for raw in steps or []:
        s = _normalize_step(raw)
        res = await execution_engine.execute_step(
            s["keyword"],
            s["arguments"],
            session_id,
            detail_level="minimal",
            assign_to=s.get("assign_to"),
            use_context=True,
        )
        results.append(res)
        if not res.get("success", False) and (
            stop_on_failure is True
            or str(stop_on_failure).lower() in ("1", "true", "yes", "on")
        ):
            break
    return results


@mcp.tool(enabled=False)
async def evaluate_expression(
    session_id: str,
    expression: str,
    assign_to: str | None = None,
) -> Dict[str, Any]:
    """Evaluate a Python expression in RF context (BuiltIn.Evaluate).

    - Uses the current RF session variables; supports ${var} inside the expression.
    - Optionally assigns the result to a variable name (test scope).
    """
    res = await execution_engine.execute_step(
        "Evaluate",
        [expression],
        session_id,
        detail_level="minimal",
        assign_to=assign_to,
        use_context=True,
    )
    return res


@mcp.tool(enabled=False)
async def set_variables(
    session_id: str,
    variables: Dict[str, Any] | List[str],
    scope: str = "test",
) -> Dict[str, Any]:
    """Set multiple variables in the RF session Variables store.

    - variables: either a dict {name: value} or a list of "name=value" strings.
    - scope: one of 'test', 'suite', 'global' (default 'test').
    """
    # Normalize input
    pairs: Dict[str, Any] = {}
    if isinstance(variables, dict):
        pairs = variables
    else:
        for item in variables:
            if isinstance(item, str) and "=" in item:
                n, v = item.split("=", 1)
                pairs[n.strip()] = v

    set_kw = {
        "test": "Set Test Variable",
        "suite": "Set Suite Variable",
        "global": "Set Global Variable",
    }.get(scope.lower(), "Set Test Variable")

    # If bridge configured, set in external context using client
    client = _get_external_client_if_configured()
    results: Dict[str, bool] = {}
    if client is not None:
        for name, value in pairs.items():
            try:
                resp = client.set_variable(name, value, scope=scope)
                results[name] = bool(resp.get("success"))
            except Exception:
                results[name] = False
        return {
            "success": all(results.values()),
            "session_id": session_id,
            "set": list(results.keys()),
            "scope": scope,
            "external": True,
        }
    for name, value in pairs.items():
        # Use RF keyword so scoping is honored
        res = await execution_engine.execute_step(
            set_kw,
            [f"${{{name}}}", value],
            session_id,
            detail_level="minimal",
            use_context=True,
        )
        results[name] = bool(res.get("success"))

    return {
        "success": all(results.values()),
        "session_id": session_id,
        "set": list(results.keys()),
        "scope": scope,
    }


async def _execute_if_impl(
    session_id: str,
    condition: str,
    then_steps: List[Dict[str, Any]],
    else_steps: List[Dict[str, Any]] | None = None,
    stop_on_failure: bool = True,
) -> Dict[str, Any]:
    """Evaluate a condition in RF context and run then/else blocks of steps."""
    # Record flow block
    try:
        sess = execution_engine.session_manager.get_or_create_session(session_id)
        block = {
            "type": "if",
            "condition": condition,
            "then": [_normalize_step(s) for s in (then_steps or [])],
            "else": [_normalize_step(s) for s in (else_steps or [])],
        }
        # ADR-005: route flow block to current test in multi-test mode
        if sess.test_registry.is_multi_test_mode():
            current = sess.test_registry.get_current_test()
            if current:
                current.flow_blocks.append(block)
            else:
                sess.flow_blocks.append(block)
        else:
            sess.flow_blocks.append(block)
    except Exception:
        pass

    cond = await execution_engine.execute_step(
        "Evaluate",
        [condition],
        session_id,
        detail_level="minimal",
        use_context=True,
    )
    truthy = False
    if cond.get("success"):
        out = str(cond.get("output", "")).strip().lower()
        truthy = out in ("true", "1", "yes", "on")
    branch = then_steps if truthy else (else_steps or [])
    step_results = await _run_steps_in_context(session_id, branch, stop_on_failure)
    ok = all(sr.get("success", False) for sr in step_results)
    return {
        "success": ok,
        "branch_taken": "then" if truthy else "else",
        "condition_result": cond.get("output") if cond.get("success") else None,
        "steps": step_results,
    }


async def _execute_for_each_impl(
    session_id: str,
    items: List[Any] | None,
    steps: List[Dict[str, Any]],
    item_var: str = "item",
    stop_on_failure: bool = True,
    max_iterations: int = 1000,
) -> Dict[str, Any]:
    """Run a sequence of steps for each item, setting ${item_var} in RF context per iteration."""
    # Record flow block (do not unroll items)
    try:
        sess = execution_engine.session_manager.get_or_create_session(session_id)
        block = {
            "type": "for_each",
            "item_var": item_var,
            "items": list(items or []),
            "body": [_normalize_step(s) for s in (steps or [])],
        }
        # ADR-005: route flow block to current test in multi-test mode
        if sess.test_registry.is_multi_test_mode():
            current = sess.test_registry.get_current_test()
            if current:
                current.flow_blocks.append(block)
            else:
                sess.flow_blocks.append(block)
        else:
            sess.flow_blocks.append(block)
    except Exception:
        pass

    if not items:
        return {"success": True, "iterations": [], "count": 0}

    iterations: List[Dict[str, Any]] = []
    count = 0
    for idx, it in enumerate(items):
        if idx >= int(max_iterations):
            break
        # Set ${item_var} in test scope using BuiltIn keyword
        _ = await execution_engine.execute_step(
            "Set Test Variable",
            [f"${{{item_var}}}", it],
            session_id,
            detail_level="minimal",
            use_context=True,
        )
        step_results = await _run_steps_in_context(session_id, steps, stop_on_failure)
        iterations.append({"index": idx, "item": it, "steps": step_results})
        count += 1
        if any(not sr.get("success", False) for sr in step_results) and stop_on_failure:
            break

    overall_success = all(
        all(sr.get("success", False) for sr in it["steps"]) for it in iterations
    )
    return {"success": overall_success, "iterations": iterations, "count": count}


async def _execute_try_except_impl(
    session_id: str,
    try_steps: List[Dict[str, Any]],
    except_patterns: List[str] | None = None,
    except_steps: List[Dict[str, Any]] | None = None,
    finally_steps: List[Dict[str, Any]] | None = None,
    rethrow: bool = False,
) -> Dict[str, Any]:
    """Execute steps in a TRY/EXCEPT/FINALLY structure."""
    # Record flow block
    try:
        sess = execution_engine.session_manager.get_or_create_session(session_id)
        block = {
            "type": "try",
            "try": [_normalize_step(s) for s in (try_steps or [])],
            "except_patterns": list(except_patterns or []),
            "except": [_normalize_step(s) for s in (except_steps or [])]
            if except_steps
            else [],
            "finally": [_normalize_step(s) for s in (finally_steps or [])]
            if finally_steps
            else [],
        }
        # ADR-005: route flow block to current test in multi-test mode
        if sess.test_registry.is_multi_test_mode():
            current = sess.test_registry.get_current_test()
            if current:
                current.flow_blocks.append(block)
            else:
                sess.flow_blocks.append(block)
        else:
            sess.flow_blocks.append(block)
    except Exception:
        pass

    # Stop try body at first failure (subsequent steps should not execute)
    try_res = await _run_steps_in_context(session_id, try_steps, stop_on_failure=True)
    first_fail = next((r for r in try_res if not r.get("success", False)), None)
    handled = False
    exc_res: List[Dict[str, Any]] | None = None
    fin_res: List[Dict[str, Any]] | None = None
    err_text = None

    if first_fail is not None:
        err_text = first_fail.get("error") or str(first_fail)
        pats = except_patterns or []
        # Glob-style match; '*' catches all
        match = False
        if not pats:
            match = True
        else:
            try:
                from fnmatch import fnmatch

                for p in pats:
                    if isinstance(p, str):
                        pat = p.strip()
                        if (
                            pat == "*"
                            or fnmatch(err_text.lower(), pat.lower())
                            or (pat.lower() in err_text.lower())
                        ):
                            match = True
                            break
            except Exception:
                match = any(
                    (isinstance(p, str) and p.lower() in err_text.lower()) for p in pats
                )
        if match and (except_steps or []):
            exc_res = await _run_steps_in_context(
                session_id, except_steps or [], stop_on_failure=False
            )
            handled = True

    if finally_steps:
        fin_res = await _run_steps_in_context(
            session_id, finally_steps, stop_on_failure=False
        )

    success = first_fail is None or handled
    result: Dict[str, Any] = {
        "success": success
        if not bool(rethrow)
        else False
        if (first_fail and not handled)
        else success,
        "handled": handled,
        "try_results": try_res,
    }
    if exc_res is not None:
        result["except_results"] = exc_res
    if fin_res is not None:
        result["finally_results"] = fin_res
    if err_text is not None and not handled:
        result["error"] = err_text
    return result


@mcp.tool(enabled=False)
async def execute_if(
    session_id: str,
    condition: str,
    then_steps: List[Dict[str, Any]],
    else_steps: List[Dict[str, Any]] | None = None,
    stop_on_failure: bool = True,
) -> Dict[str, Any]:
    return await _execute_if_impl(
        session_id=session_id,
        condition=condition,
        then_steps=then_steps,
        else_steps=else_steps,
        stop_on_failure=stop_on_failure,
    )


@mcp.tool(enabled=False)
async def execute_for_each(
    session_id: str,
    items: List[Any] | None,
    steps: List[Dict[str, Any]],
    item_var: str = "item",
    stop_on_failure: bool = True,
    max_iterations: int = 1000,
) -> Dict[str, Any]:
    return await _execute_for_each_impl(
        session_id=session_id,
        items=items,
        steps=steps,
        item_var=item_var,
        stop_on_failure=stop_on_failure,
        max_iterations=max_iterations,
    )


@mcp.tool(enabled=False)
async def execute_try_except(
    session_id: str,
    try_steps: List[Dict[str, Any]],
    except_patterns: List[str] | None = None,
    except_steps: List[Dict[str, Any]] | None = None,
    finally_steps: List[Dict[str, Any]] | None = None,
    rethrow: bool = False,
) -> Dict[str, Any]:
    return await _execute_try_except_impl(
        session_id=session_id,
        try_steps=try_steps,
        except_patterns=except_patterns,
        except_steps=except_steps,
        finally_steps=finally_steps,
        rethrow=rethrow,
    )


@mcp.tool
async def get_keyword_info(
    mode: str = "keyword",
    keyword_name: str | None = None,
    library_name: str | None = None,
    session_id: str | None = None,
    arguments: List[str] | None = None,
) -> Dict[str, Any]:
    """Retrieve keyword or library documentation, or parse signatures.

    Args:
        mode: One of "keyword" (default), "library", "session", or "parse".
        keyword_name: Keyword to document (required for modes "keyword"/"session"/"parse").
        library_name: Library to document (required for mode "library"; optional for keyword mode).
        session_id: Session id when mode="session" to fetch overrides from the live namespace.
        arguments: Optional arguments to parse when mode="parse".

    Returns:
        Dict[str, Any]: Documentation or parse payload:
            - success: bool
            - mode: resolved mode
            - doc/signature data or error on failure
    """

    mode_norm = (mode or "keyword").strip().lower()

    if mode_norm in {"keyword", "global"}:
        if not keyword_name:
            return {"success": False, "error": "keyword_name is required"}
        result = await _get_keyword_documentation_payload(keyword_name, library_name)
        result["mode"] = "keyword"
        return result

    if mode_norm in {"library", "libdoc"}:
        if not library_name:
            return {"success": False, "error": "library_name is required"}
        result = await _get_library_documentation_payload(library_name)
        result["mode"] = "library"
        return result

    if mode_norm in {"session", "namespace"}:
        if not session_id or not keyword_name:
            return {
                "success": False,
                "error": "session_id and keyword_name are required for mode='session'",
            }
        result = await _get_session_keyword_documentation_payload(
            session_id, keyword_name
        )
        result["mode"] = "session"
        return result

    if mode_norm in {"parse", "signature"}:
        if not keyword_name:
            return {"success": False, "error": "keyword_name is required"}
        parsed = await _debug_parse_keyword_arguments_payload(
            keyword_name=keyword_name,
            arguments=arguments or [],
            library_name=library_name,
            session_id=session_id,
        )
        parsed["mode"] = "parse"
        return parsed

    return {"success": False, "error": f"Unsupported mode '{mode}'"}


async def _get_keyword_documentation_payload(
    keyword_name: str, library_name: str | None = None
) -> Dict[str, Any]:
    return execution_engine.get_keyword_documentation(keyword_name, library_name)


@mcp.tool(enabled=False)
async def get_keyword_documentation(
    keyword_name: str, library_name: str = None
) -> Dict[str, Any]:
    """Get full documentation for a specific Robot Framework keyword using native RF libdoc.

    Uses Robot Framework's native LibraryDocumentation and KeywordDoc objects to provide
    comprehensive keyword information including source location, argument types, and
    deprecation status when available.

    Args:
        keyword_name: Name of the keyword to get documentation for
        library_name: Optional library name to narrow search

    Returns:
        Dict containing comprehensive keyword information:
        - success: Boolean indicating if keyword was found
        - keyword: Dict with keyword details including:
          - name, library, args: Basic keyword information
          - arg_types: Argument types from libdoc (when available)
          - doc: Full documentation text
          - short_doc: Native Robot Framework short_doc
          - tags: Keyword tags
          - is_deprecated: Deprecation status (libdoc only)
          - source: Source file path (libdoc only)
          - lineno: Line number in source (libdoc only)
    """
    return await _get_keyword_documentation_payload(keyword_name, library_name)


async def _get_library_documentation_payload(library_name: str) -> Dict[str, Any]:
    return execution_engine.get_library_documentation(library_name)


@mcp.tool(enabled=False)
async def get_library_documentation(library_name: str) -> Dict[str, Any]:
    """Get full documentation for a Robot Framework library using native RF libdoc.

    Uses Robot Framework's native LibraryDocumentation API to provide comprehensive
    library information including library metadata and all keywords with their
    documentation, arguments, and metadata.

    Args:
        library_name: Name of the library to get documentation for

    Returns:
        Dict containing comprehensive library information:
        - success: Boolean indicating if library was found
        - library: Dict with library details including:
          - name: Library name
          - doc: Library documentation
          - version: Library version
          - type: Library type
          - scope: Library scope
          - source: Source file path
          - keywords: List of all library keywords with full details including:
            - name: Keyword name
            - args: List of argument names
            - arg_types: List of argument types (when available from libdoc)
            - doc: Full keyword documentation text
            - short_doc: Native Robot Framework short_doc
            - tags: Keyword tags
            - is_deprecated: Deprecation status (libdoc only)
            - source: Source file path (libdoc only)
            - lineno: Line number in source (libdoc only)
          - keyword_count: Total number of keywords in library
          - data_source: 'libdoc' or 'inspection' indicating data source
    """
    return await _get_library_documentation_payload(library_name)


async def _debug_parse_keyword_arguments_payload(
    keyword_name: str,
    arguments: List[str],
    library_name: str = None,
    session_id: str = None,
) -> Dict[str, Any]:
    """Debug helper: Parse arguments into positional and named using RF-native logic.

    Uses the same parsing path as execution to verify how name=value pairs are handled
    for a given keyword and (optionally) library.

    Args:
        keyword_name: Keyword to parse for (e.g., 'Open Application').
        arguments: List of argument strings as they would be passed to execute_step.
        library_name: Optional library name to disambiguate (e.g., 'AppiumLibrary').
        session_id: Optional session to pull variables from for resolution.

    Returns:
        - success: True
        - parsed: { positional: [...], named: {k: v} }
        - notes: brief info on library and session impact
    """
    try:
        session_vars = {}
        if session_id:
            sess = execution_engine.get_session(session_id)
            if sess:
                session_vars = sess.variables

        parsed = execution_engine.keyword_executor.argument_processor.parse_arguments_for_keyword(
            keyword_name, arguments, library_name, session_vars
        )
        return {
            "success": True,
            "parsed": {"positional": parsed.positional, "named": parsed.named},
            "notes": {
                "library_name": library_name,
                "session_id": session_id,
                "positional_count": len(parsed.positional),
                "named_count": len(parsed.named or {}),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(enabled=False)
async def debug_parse_keyword_arguments(
    keyword_name: str,
    arguments: List[str],
    library_name: str = None,
    session_id: str = None,
) -> Dict[str, Any]:
    return await _debug_parse_keyword_arguments_payload(
        keyword_name=keyword_name,
        arguments=arguments,
        library_name=library_name,
        session_id=session_id,
    )


# TOOL DISABLED: validate_step_before_suite
#
# Reason for removal: This tool is functionally redundant with execute_step().
# Analysis shows that it duplicates execution (performance impact) and adds
# minimal unique value beyond what execute_step() already provides.
#
# Key issues:
# 1. Functional redundancy - re-executes the same step as execute_step()
# 2. Performance overhead - double execution of steps
# 3. Agent confusion - two similar tools with overlapping purposes
# 4. Limited additional value - only adds guidance text and redundant metadata
#
# The validation workflow can be achieved with:
# execute_step() → validate_test_readiness() → build_test_suite()
#
# @mcp.tool
# async def validate_step_before_suite(
#     keyword: str,
#     arguments: List[str] = None,
#     session_id: str = "default",
#     expected_outcome: str = None,
# ) -> Dict[str, Any]:
#     """Validate a single step before adding it to a test suite.
#
#     This method enforces stepwise test development by requiring step validation
#     before suite generation. Use this to verify each keyword works as expected.
#
#     Workflow:
#     1. Call this method for each test step
#     2. Verify the step succeeds and produces expected results
#     3. Only after all steps are validated, use build_test_suite()
#
#     Args:
#         keyword: Robot Framework keyword to validate
#         arguments: Arguments for the keyword
#         session_id: Session identifier
#         expected_outcome: Optional description of expected result for validation
#
#     Returns:
#         Validation result with success status, output, and recommendations
#     """
#     if arguments is None:
#         arguments = []
#
#     # Execute the step with detailed error reporting
#     result = await execution_engine.execute_step(
#         keyword, arguments, session_id, detail_level="full"
#     )
#
#     # Add validation metadata
#     result["validated"] = result.get("success", False)
#     result["validation_time"] = result.get("execution_time")
#
#     if expected_outcome:
#         result["expected_outcome"] = expected_outcome
#         result["meets_expectation"] = "unknown"  # AI agent should evaluate this
#
#     # Add guidance for next steps
#     if result.get("success"):
#         result["next_step_guidance"] = (
#             "✅ Step validated successfully. Safe to include in test suite."
#         )
#     else:
#         result["next_step_guidance"] = (
#             "❌ Step failed validation. Fix issues before adding to test suite."
#         )
#         result["debug_suggestions"] = [
#             "Check keyword spelling and library availability",
#             "Verify argument types and values",
#             "Ensure required browser/context is open",
#             "Review error message for specific issues",
#         ]
#
#     return result


async def _get_session_validation_status_payload(
    session_id: str = "",
) -> Dict[str, Any]:
    # Import session resolver here to avoid circular imports
    from robotmcp.utils.session_resolution import SessionResolver

    session_resolver = SessionResolver(execution_engine.session_manager)

    # Resolve session with intelligent fallback
    resolution_result = session_resolver.resolve_session_with_fallback(session_id)

    if not resolution_result["success"]:
        # Return enhanced error with guidance
        return {
            "success": False,
            "error": f"Session '{session_id}' not found",
            "error_details": resolution_result["error_guidance"],
            "available_sessions": resolution_result["error_guidance"][
                "available_sessions"
            ],
            "sessions_with_steps": resolution_result["error_guidance"][
                "sessions_with_steps"
            ],
            "recommendation": "Use analyze_scenario() to create a session first",
        }

    # Use resolved session ID
    resolved_session_id = resolution_result["session_id"]

    # Get validation status for resolved session
    result = execution_engine.get_session_validation_status(resolved_session_id)

    # Add session resolution info to result
    if resolution_result.get("fallback_used", False):
        result["session_resolution"] = {
            "fallback_used": True,
            "original_session_id": session_id,
            "resolved_session_id": resolved_session_id,
            "message": f"Automatically checked session '{resolved_session_id}'",
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id,
        }

    return result


@mcp.tool(enabled=False)
async def validate_test_readiness(session_id: str = "default") -> Dict[str, Any]:
    """Check if session is ready for test suite generation.

    Enforces stepwise workflow by verifying all steps have been validated.
    Use this before calling build_test_suite() to ensure quality.

    Args:
        session_id: Session identifier to validate

    Returns:
        Readiness status with guidance on next actions
    """
    return await execution_engine.validate_test_readiness(session_id)


@mcp.tool
async def set_library_search_order(
    libraries: List[str], session_id: str = "default"
) -> Dict[str, Any]:
    """Set explicit library search order for keyword resolution.

    Args:
        libraries: Library names in priority order (highest first).
        session_id: Session to apply the search order to.

    Returns:
        Dict[str, Any]: Result payload:
            - success: bool
            - session_id: echoed id
            - old_search_order/new_search_order: before/after lists
            - warnings: any invalid or missing libraries
    """
    try:
        # Get or create session
        session = execution_engine.session_manager.get_or_create_session(session_id)

        # Set library search order
        old_order = session.get_search_order()
        session.set_library_search_order(libraries)
        new_order = session.get_search_order()

        return {
            "success": True,
            "session_id": session_id,
            "old_search_order": old_order,
            "new_search_order": new_order,
            "libraries_requested": libraries,
            "libraries_applied": new_order,
            "message": f"Library search order updated for session '{session_id}'",
        }

    except Exception as e:
        logger.error(f"Error setting library search order: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


@mcp.tool(enabled=False)
async def initialize_context(
    session_id: str, libraries: List[str] = None, variables: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Initialize a session with libraries and variables.

    NOTE: Full RF context mode is not yet implemented. This tool currently
    initializes a session with the specified libraries and variables using
    the existing session-based variable system.

    Args:
        session_id: Session identifier
        libraries: List of libraries to import in the session
        variables: Initial variables to set in the session

    Returns:
        Session initialization status with information
    """
    try:
        # Get or create session
        session = execution_engine.session_manager.get_or_create_session(session_id)

        # Import libraries into session
        if libraries:
            for library in libraries:
                try:
                    session.import_library(library)
                    # Also add to loaded_libraries for tracking
                    session.loaded_libraries.add(library)
                    logger.info(f"Imported {library} into session {session_id}")
                except Exception as lib_error:
                    logger.warning(f"Could not import {library}: {lib_error}")

        # Set initial variables in session
        if variables:
            for name, value in variables.items():
                # Normalize variable name to RF format
                if not name.startswith("$"):
                    var_name = f"${{{name}}}"
                else:
                    var_name = name
                session.set_variable(var_name, value)
                logger.info(
                    f"Set variable {var_name} = {value} in session {session_id}"
                )

        return {
            "success": True,
            "session_id": session_id,
            "context_enabled": False,  # Context mode not fully implemented
            "libraries_loaded": list(session.loaded_libraries),
            "variables_set": list(variables.keys()) if variables else [],
            "message": f"Session '{session_id}' initialized with libraries and variables",
            "note": "Using session-based variable system (context mode not available)",
        }

    except Exception as e:
        logger.error(f"Error initializing session {session_id}: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


async def _get_context_variables_payload(session_id: str) -> Dict[str, Any]:
    """Get variables, preferring bridge when available.

    This function implements Phase 4 of ADR-003 Debug Bridge Unification:
    - Uses _get_external_client_with_session_sync to auto-create session from bridge
    - Ensures first-call success when bridge is active with RF context
    """
    try:
        # Helper to sanitize values: return scalars as-is; for complex objects, return their type name.
        def _sanitize(val: Any) -> Any:
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            # Avoid serializing complex/large objects
            return f"<{type(val).__name__}>"

        # --- Phase 4: Use unified client+session retrieval with auto-sync ---
        # This ensures sessions are auto-created from bridge state when needed,
        # fixing first-call failures when bridge is active.
        client, session = _get_external_client_with_session_sync(session_id)

        # --- Attach bridge routing (R3) ---
        # When an attach bridge is configured, read variables from the live RF
        # process.  This ensures variables created during RF execution (e.g. by
        # test setup, resource files, keyword returns) are visible to MCP.
        if client is not None:
            try:
                bridge_resp = client.get_variables()
                if bridge_resp.get("success"):
                    raw_vars = bridge_resp.get("result", {})
                    sanitized = {}
                    for k, v in raw_vars.items():
                        key = k
                        if key.startswith("${") and key.endswith("}"):
                            key = key[2:-1]
                        sanitized[key] = _sanitize(v)
                    return {
                        "success": True,
                        "session_id": session_id,
                        "variables": sanitized,
                        "variable_count": len(sanitized),
                        "source": "attach_bridge",
                        "truncated": bridge_resp.get("truncated", False),
                    }
            except Exception as bridge_err:
                logger.debug(
                    f"Attach bridge variable read failed, falling back: {bridge_err}"
                )

        # Prefer RF Namespace/Variables if an RF context exists for the session
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            ctx_info = mgr.get_session_context_info(session_id)
            if ctx_info.get("context_exists"):
                # Extract variables from RF Variables object
                ctx = mgr._session_contexts.get(session_id)  # internal read-only access
                rf_vars_obj = ctx.get("variables") if ctx else None
                rf_vars: Dict[str, Any] = {}
                if rf_vars_obj is not None:
                    try:
                        if hasattr(rf_vars_obj, "store"):
                            rf_vars = dict(rf_vars_obj.store.data)
                        elif hasattr(rf_vars_obj, "current") and hasattr(
                            rf_vars_obj.current, "store"
                        ):
                            rf_vars = dict(rf_vars_obj.current.store.data)
                    except Exception:
                        rf_vars = {}

                # Attempt to resolve variable resolvers to concrete values via Variables API
                resolved: Dict[str, Any] = {}
                for k, v in rf_vars.items():
                    key = k if isinstance(k, str) else str(k)
                    try:
                        norm = key if key.startswith("${") else f"${{{key}}}"
                        concrete = rf_vars_obj[norm]
                    except Exception:
                        concrete = v
                    resolved[key if not key.startswith("${") else key.strip("${}")] = (
                        concrete
                    )

                sanitized = {str(k): _sanitize(v) for k, v in resolved.items()}
                return {
                    "success": True,
                    "session_id": session_id,
                    "variables": sanitized,
                    "variable_count": len(sanitized),
                    "source": "rf_context",
                }
        except Exception:
            # Fall back to session store below
            pass

        # Fallback: session-based variable store
        # Note: session may have been auto-created by _get_external_client_with_session_sync
        if not session:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found and no active attach bridge",
                "session_id": session_id,
                "hint": "Call execute_step or manage_session(action='init') first, or configure attach bridge",
            }
        sess_vars_raw = dict(session.variables)
        sess_vars = {str(k): _sanitize(v) for k, v in sess_vars_raw.items()}
        return {
            "success": True,
            "session_id": session_id,
            "variables": sess_vars,
            "variable_count": len(sess_vars),
            "source": "session_store",
        }

    except Exception as e:
        logger.error(f"Error getting variables for session {session_id}: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


@mcp.tool(enabled=False)
async def get_context_variables(session_id: str) -> Dict[str, Any]:
    """Get all variables from a session."""
    return await _get_context_variables_payload(session_id)


async def _get_session_info_payload(session_id: str = "default") -> Dict[str, Any]:
    """Get session info, auto-initializing from bridge if needed.

    This function implements Phase 1 of ADR-003 Debug Bridge Unification:
    - Uses _get_external_client_with_session_sync() to auto-create session from bridge
    - Does not fail if bridge is available even without local session
    """
    try:
        # Use unified client+session retrieval (auto-creates session from bridge if needed)
        client, session = _get_external_client_with_session_sync(session_id)
        base_info = session.get_session_info() if session else {}

        # S5: Enrich with bridge data when available
        if client is not None:
            try:
                diag = client.diagnostics()
                if diag.get("success"):
                    bridge_result = diag.get("result", {})
                    base_info["attach_bridge"] = {
                        "active": True,
                        "libraries": bridge_result.get("libraries", []),
                        "context_active": bridge_result.get("context", False),
                    }
            except Exception as bridge_err:
                logger.debug(
                    f"Bridge diagnostics for session info failed: {bridge_err}"
                )
                base_info["attach_bridge"] = {"active": False, "error": "unreachable"}

        # CRITICAL FIX (ADR-003 S1): Don't fail if bridge is available
        if not session and not base_info.get("attach_bridge", {}).get("active"):
            return {
                "success": False,
                "error": f"Session '{session_id}' not found and no active attach bridge",
                "available_sessions": execution_engine.session_manager.get_all_session_ids(),
                "hint": "Call execute_step or manage_session(action='init') first, or configure attach bridge",
            }

        # ADR-005: Expose multi-test info when active
        if session and session.test_registry.is_multi_test_mode():
            tests_info = []
            for name, ti in session.test_registry.tests.items():
                tests_info.append({
                    "name": name,
                    "status": ti.status,
                    "step_count": len(ti.steps),
                    "tags": ti.tags,
                })
            base_info["multi_test_mode"] = True
            base_info["current_test"] = session.test_registry.current_test_name
            base_info["tests"] = tests_info
            if session.suite_setup:
                base_info["suite_setup"] = session.suite_setup
            if session.suite_teardown:
                base_info["suite_teardown"] = session.suite_teardown
            base_info["suite_level_step_count"] = len(session.suite_level_steps)

        return {"success": True, "session_info": base_info}

    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


@mcp.tool(enabled=False)
async def get_session_info(session_id: str = "default") -> Dict[str, Any]:
    """Get comprehensive information about a session's configuration and state."""
    return await _get_session_info_payload(session_id)


@mcp.tool
async def get_locator_guidance(
    library: str = "browser",
    error_message: str | None = None,
    keyword_name: str | None = None,
) -> Dict[str, Any]:
    """Provide locator/selector guidance for Browser, SeleniumLibrary, or AppiumLibrary.

    Args:
        library: Target library ("Browser", "SeleniumLibrary", or "AppiumLibrary"). Case-insensitive.
        error_message: Optional error text to tailor guidance (e.g., from a failed keyword).
        keyword_name: Optional keyword name for context-specific hints.

    Returns:
        Dict[str, Any]: Guidance payload:
            - success: bool
            - library: resolved library name
            - tips/warnings/examples: library-specific suggestions
            - error: present when library is unsupported
    """

    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    lib_norm = (library or "browser").strip().lower()

    if lib_norm in {"browser", "playwright"}:
        result = converter.get_browser_locator_guidance(error_message, keyword_name)
        result["library"] = "Browser"
        result.setdefault("success", True)
        return result

    if lib_norm in {"selenium", "seleniumlibrary"}:
        result = converter.get_selenium_locator_guidance(error_message, keyword_name)
        result["library"] = "SeleniumLibrary"
        result.setdefault("success", True)
        return result

    if lib_norm in {"appium", "appiumlibrary"}:
        result = converter.get_appium_locator_guidance(error_message, keyword_name)
        result["library"] = "AppiumLibrary"
        result.setdefault("success", True)
        return result

    return {
        "success": False,
        "error": f"Unsupported library '{library}'. Choose Browser, SeleniumLibrary, or AppiumLibrary.",
    }


@mcp.tool(enabled=False)
async def get_selenium_locator_guidance(
    error_message: str = None, keyword_name: str = None
) -> Dict[str, Any]:
    """Get comprehensive SeleniumLibrary locator strategy guidance for AI agents.

    This tool helps AI agents understand SeleniumLibrary's locator strategies and
    provides context-aware suggestions for element location and error resolution.

    SeleniumLibrary supports these locator strategies:
    - id: Element id (e.g., 'id:example')
    - name: name attribute (e.g., 'name:example')
    - identifier: Either id or name (e.g., 'identifier:example')
    - class: Element class (e.g., 'class:example')
    - tag: Tag name (e.g., 'tag:div')
    - xpath: XPath expression (e.g., 'xpath://div[@id="example"]')
    - css: CSS selector (e.g., 'css:div#example')
    - dom: DOM expression (e.g., 'dom:document.images[5]')
    - link: Exact link text (e.g., 'link:Click Here')
    - partial link: Partial link text (e.g., 'partial link:Click')
    - data: Element data-* attribute (e.g., 'data:id:my_id')
    - jquery: jQuery expression (e.g., 'jquery:div.example')
    - default: Keyword-specific default (e.g., 'default:example')

    Args:
        error_message: Optional error message to analyze for specific guidance
        keyword_name: Optional keyword name that failed for context-specific tips

    Returns:
        Comprehensive locator strategy guidance with examples, tips, and error-specific advice
    """
    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    return converter.get_selenium_locator_guidance(error_message, keyword_name)


@mcp.tool(enabled=False)
async def get_browser_locator_guidance(
    error_message: str = None, keyword_name: str = None
) -> Dict[str, Any]:
    """Get comprehensive Browser Library (Playwright) locator strategy guidance for AI agents.

    This tool helps AI agents understand Browser Library's selector strategies and
    provides context-aware suggestions for element location and error resolution.

    Browser Library uses Playwright's locator strategies with these key features:

    **Selector Strategies:**
    - css: CSS selector (default) - e.g., '.button' or 'css=.button'
    - xpath: XPath expression - e.g., '//button' or 'xpath=//button'
    - text: Text content matching - e.g., '"Login"' or 'text=Login'
    - id: Element ID - e.g., 'id=submit-btn'
    - data-testid: Test ID attribute - e.g., 'data-testid=login-button'

    **Advanced Features:**
    - Cascaded selectors: 'text=Hello >> ../.. >> .select_button'
    - iFrame piercing: 'id=myframe >>> .inner-button'
    - Shadow DOM: Automatic piercing with CSS and text engines
    - Strict mode: Controls behavior with multiple element matches
    - Element references: '${ref} >> .child' for chained operations

    **Implicit Detection Rules:**
    - Plain selectors → CSS (default): '.button' becomes 'css=.button'
    - Starting with // or .. → XPath: '//button' becomes 'xpath=//button'
    - Quoted text → Text selector: '"Login"' becomes 'text=Login'
    - Explicit format: 'strategy=value' for any strategy

    Args:
        error_message: Optional error message to analyze for specific guidance
        keyword_name: Optional keyword name that failed for context-specific tips

    Returns:
        Comprehensive Browser Library locator guidance with examples, patterns, and error-specific advice
    """
    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    return converter.get_browser_locator_guidance(error_message, keyword_name)


@mcp.tool(enabled=False)
async def get_appium_locator_guidance(
    error_message: str = None, keyword_name: str = None
) -> Dict[str, Any]:
    """Get comprehensive AppiumLibrary locator strategy guidance for AI agents.

    This tool helps AI agents understand AppiumLibrary's locator strategies and
    provides context-aware suggestions for mobile element location and error resolution.

    AppiumLibrary supports these locator strategies:

    **Basic Locators:**
    - id: Element ID (e.g., 'id=my_element' or just 'my_element')
    - xpath: XPath expression (e.g., '//*[@type="android.widget.EditText"]')
    - identifier: Matches by @id attribute (e.g., 'identifier=my_element')
    - accessibility_id: Accessibility options utilize (e.g., 'accessibility_id=button3')
    - class: Matches by class (e.g., 'class=UIAPickerWheel')
    - name: Matches by @name attribute (e.g., 'name=my_element') - Only valid for Selendroid

    **Platform-Specific Locators:**
    - android: Android UI Automator (e.g., 'android=UiSelector().description("Apps")')
    - ios: iOS UI Automation (e.g., 'ios=.buttons().withName("Apps")')
    - predicate: iOS Predicate (e.g., 'predicate=name=="login"')
    - chain: iOS Class Chain (e.g., 'chain=XCUIElementTypeWindow[1]/*')

    **WebView Locators:**
    - css: CSS selector in webview (e.g., 'css=.green_button')

    **Default Behavior:**
    - By default, locators match against key attributes (id for all elements)
    - Plain text (e.g., 'my_element') is treated as ID lookup
    - XPath should start with // or use explicit 'xpath=' prefix

    **WebElement Support:**
    Starting with AppiumLibrary v1.4, you can pass WebElement objects:
    - Get elements with: Get WebElements or Get WebElement
    - Use directly: Click Element ${element}

    Args:
        error_message: Optional error message to analyze for specific guidance
        keyword_name: Optional keyword name that failed for context-specific tips

    Returns:
        Comprehensive locator strategy guidance with examples, tips, and error-specific advice
    """
    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    return converter.get_appium_locator_guidance(error_message, keyword_name)


async def _get_loaded_libraries_payload() -> Dict[str, Any]:
    # S4: Check bridge first and use its library list when available
    client = _get_external_client_if_configured()
    if client is not None:
        try:
            diag = client.diagnostics()
            if diag.get("success"):
                bridge_libs = diag.get("result", {}).get("libraries", [])
                return {
                    "success": True,
                    "source": "attach_bridge",
                    "libraries": bridge_libs,
                    "library_count": len(bridge_libs),
                    "context_active": diag.get("result", {}).get("context", False),
                }
        except Exception as e:
            logger.debug(f"Library status via bridge failed: {e}")

    return execution_engine.get_library_status()


@mcp.tool(enabled=False)
async def get_loaded_libraries() -> Dict[str, Any]:
    """Get status of all loaded Robot Framework libraries using both libdoc and inspection methods."""
    return await _get_loaded_libraries_payload()


@mcp.tool(enabled=False)
async def run_test_suite_dry(
    session_id: str = "",
    suite_file_path: str = None,
    validation_level: str = "standard",
    include_warnings: bool = True,
) -> Dict[str, Any]:
    """Validate test suite using Robot Framework dry run mode.

    RECOMMENDED WORKFLOW - SUITE VALIDATION:
    This tool should be used AFTER build_test_suite to validate the generated suite:
    1. ✅ build_test_suite - Generate .robot file from session steps
    2. ✅ run_test_suite_dry (THIS TOOL) - Validate syntax and structure
    3. ➡️ run_test_suite - Execute if validation passes

    Enhanced Session Resolution:
    - If session_id provided and valid: Uses that session's generated suite
    - If session_id empty/invalid: Automatically finds most suitable session
    - If suite_file_path provided: Validates specified file directly

    Validation Levels:
    - minimal: Basic syntax checking only
    - standard: Syntax + keyword verification + imports (default)
    - strict: All checks + argument validation + structure analysis

    Args:
        session_id: Session with executed steps (auto-resolves if empty/invalid)
        suite_file_path: Direct path to .robot file (optional, overrides session)
        validation_level: Validation depth ('minimal', 'standard', 'strict')
        include_warnings: Include warnings in validation report

    Returns:
        Structured validation results with issues, warnings, and suggestions
    """

    # Session resolution with same logic as build_test_suite
    from robotmcp.utils.session_resolution import SessionResolver

    session_resolver = SessionResolver(execution_engine.session_manager)

    if suite_file_path:
        # Direct file validation mode
        logger.info(f"Running dry run validation on file: {suite_file_path}")
        return await execution_engine.run_suite_dry_run_from_file(
            suite_file_path, validation_level, include_warnings
        )
    else:
        # Session-based validation mode
        resolution_result = session_resolver.resolve_session_with_fallback(session_id)

        if not resolution_result["success"]:
            return {
                "success": False,
                "tool": "run_test_suite_dry",
                "error": "No valid session or suite file for validation",
                "error_details": resolution_result["error_guidance"],
                "guidance": [
                    "Create a session and execute some steps first",
                    "Use build_test_suite to generate a test suite",
                    "Or provide suite_file_path to validate an existing file",
                ],
                "recommendation": "Use build_test_suite first or provide suite_file_path",
            }

        resolved_session_id = resolution_result["session_id"]
        logger.info(f"Running dry run validation for session: {resolved_session_id}")

        result = await execution_engine.run_suite_dry_run(
            resolved_session_id, validation_level, include_warnings
        )

        # Add session resolution info to result
        if resolution_result.get("fallback_used", False):
            result["session_resolution"] = {
                "fallback_used": True,
                "original_session_id": session_id,
                "resolved_session_id": resolved_session_id,
                "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps",
            }
        else:
            result["session_resolution"] = {
                "fallback_used": False,
                "session_id": resolved_session_id,
            }

        return result


@mcp.tool
async def run_test_suite(
    session_id: str = "",
    suite_file_path: str = None,
    mode: str = "full",
    validation_level: str = "standard",
    include_warnings: bool = True,
    execution_options: Dict[str, Any] = None,
    output_level: str = "standard",
    capture_screenshots: bool = False,
) -> Dict[str, Any]:
    """Validate or execute a Robot Framework suite.

    Args:
        session_id: Session containing steps to build/execute; optional if suite_file_path is given.
        suite_file_path: Path to an existing .robot file to validate/execute.
        mode: "dry"/"validate" for dry run; "full" to execute. Defaults to "full".
        validation_level: Dry-run validation depth ("minimal", "standard", "strict"). Default "standard".
        include_warnings: Whether to include warnings in validation output.
        execution_options: RF execution options (variables, tags, loglevel, timeout, etc.).
        output_level: Response verbosity ("minimal", "standard", "detailed").
        capture_screenshots: Enable screenshot capture on failures (if supported).

    Returns:
        Dict[str, Any]: Suite result:
            - success: bool
            - mode: "dry" or "full"
            - statistics/execution_details/output_files when executed
            - validation_results when dry run
            - session_resolution: info when fallback session resolution is used
            - error/guidance: present on failure
    """

    if execution_options is None:
        execution_options = {}

    mode_norm = (mode or "full").strip().lower()

    from robotmcp.utils.session_resolution import SessionResolver

    session_resolver = SessionResolver(execution_engine.session_manager)

    if suite_file_path:
        if mode_norm in {"dry", "validate", "validation"}:
            logger.info(f"Running dry run validation on file: {suite_file_path}")
            result = await execution_engine.run_suite_dry_run_from_file(
                suite_file_path, validation_level, include_warnings
            )
            result["mode"] = "dry"
            return result

        logger.info(f"Running suite execution on file: {suite_file_path}")
        result = await execution_engine.run_suite_execution_from_file(
            suite_file_path, execution_options, output_level, capture_screenshots
        )
        result["mode"] = "full"
        return result

    resolution_result = session_resolver.resolve_session_with_fallback(session_id)

    if not resolution_result["success"]:
        tool_name = (
            "run_test_suite_dry"
            if mode_norm in {"dry", "validate", "validation"}
            else "run_test_suite"
        )
        return {
            "success": False,
            "tool": tool_name,
            "mode": mode_norm,
            "error": "No valid session or suite file available",
            "error_details": resolution_result["error_guidance"],
            "guidance": [
                "Create a session and execute some steps first",
                "Use build_test_suite to generate a test suite",
                "Or provide suite_file_path to validate or execute an existing file",
            ],
            "recommendation": "Use build_test_suite first or provide suite_file_path",
        }

    resolved_session_id = resolution_result["session_id"]

    if mode_norm in {"dry", "validate", "validation"}:
        logger.info(f"Running dry run validation for session: {resolved_session_id}")
        result = await execution_engine.run_suite_dry_run(
            resolved_session_id, validation_level, include_warnings
        )
        result["mode"] = "dry"
    else:
        logger.info(f"Running suite execution for session: {resolved_session_id}")
        result = await execution_engine.run_suite_execution(
            resolved_session_id, execution_options, output_level, capture_screenshots
        )
        result["mode"] = "full"

    if resolution_result.get("fallback_used", False):
        result["session_resolution"] = {
            "fallback_used": True,
            "original_session_id": session_id,
            "resolved_session_id": resolved_session_id,
            "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps",
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id,
        }

    return result


@mcp.tool(enabled=False)
async def get_session_validation_status(session_id: str = "") -> Dict[str, Any]:
    """Get validation status of all steps in a session with intelligent session resolution."""
    return await _get_session_validation_status_payload(session_id)


async def _diagnose_rf_context_payload(session_id: str) -> Dict[str, Any]:
    """Return diagnostic information about the current RF execution context for a session.

    Includes: whether context exists, created_at, imported libraries, variables count,
    and where possible, the current RF library search order.
    """
    try:
        client = _get_external_client_if_configured()
        if client is not None:
            r = client.diagnostics()
            return {
                "context_exists": r.get("success", False),
                "external": True,
                "result": r.get("result"),
            }
        mgr = get_rf_native_context_manager()
        info = mgr.get_session_context_info(session_id)
        # Try to enrich with Namespace search order and imported libraries
        if info.get("context_exists"):
            ctx = mgr._session_contexts.get(session_id)  # internal, read-only
            extra = {}
            try:
                namespace = ctx.get("namespace")
                # Namespace has no direct getter for search order; infer from libraries list
                lib_names = []
                if hasattr(namespace, "libraries"):
                    libs = namespace.libraries
                    if hasattr(libs, "keys"):
                        lib_names = list(libs.keys())
                extra["namespace_libraries"] = lib_names
            except Exception:
                pass
            info["extra"] = extra
        return info
    except Exception as e:
        logger.error(f"diagnose_rf_context failed: {e}")
        return {"context_exists": False, "error": str(e), "session_id": session_id}


@mcp.tool(
    name="diagnose_rf_context",
    description="Inspect RF context state for a session: libraries, search order, and variables count.",
    enabled=False,
)
async def diagnose_rf_context(session_id: str) -> Dict[str, Any]:
    return await _diagnose_rf_context_payload(session_id)


@mcp.tool
async def manage_attach(action: str = "status") -> Dict[str, Any]:
    """Inspect or control attach bridge configuration.

    Args:
        action: One of:
            - "status" (default): Check bridge configuration and health
            - "stop": Send stop command to bridge (sets stop flag)
            - "cleanup"/"clean": Clean expired sessions and check bridge health
            - "reset"/"reconnect": Stop bridge and clean all local sessions
            - "disconnect_all"/"terminate"/"force_stop": Force stop bridge and terminate all

    Returns:
        Dict[str, Any]: Attach status payload:
            - success: bool
            - action: echoed action
            - configured/reachable/default_mode/strict: attach configuration fields
            - diagnostics/hint/error: context-specific fields

        For cleanup action:
            - sessions_cleaned: number of expired sessions removed
            - remaining_sessions: count of active sessions
            - bridge_status: bridge health information

        For reset action:
            - bridge_stopped: whether bridge was successfully stopped
            - sessions_cleaned: number of sessions removed
            - recovery_hint: how to restart the bridge

        For disconnect_all action:
            - bridge_stopped: whether bridge was successfully stopped
            - sessions_cleaned: number of sessions removed
            - hint: next steps for bridge restart
    """

    action_norm = (action or "status").strip().lower()

    if action_norm in {"status", "info"}:
        try:
            client = _get_external_client_if_configured()
            mode = os.environ.get("ROBOTMCP_ATTACH_DEFAULT", "auto").strip().lower()
            strict = os.environ.get("ROBOTMCP_ATTACH_STRICT", "0").strip() in {
                "1",
                "true",
                "yes",
            }
            if client is None:
                return {
                    "success": True,
                    "action": "status",
                    "configured": False,
                    "default_mode": mode,
                    "strict": strict,
                    "hint": "Set ROBOTMCP_ATTACH_HOST to enable attach mode.",
                }
            diag = client.diagnostics()
            return {
                "success": True,
                "action": "status",
                "configured": True,
                "host": client.host,
                "port": client.port,
                "reachable": bool(diag.get("success")),
                "diagnostics": diag.get("result"),
                "default_mode": mode,
                "strict": strict,
                "hint": "execute_step(..., use_context=True) routes to the bridge when reachable.",
            }
        except Exception as e:
            logger.error(f"attach status failed: {e}")
            return {"success": False, "action": "status", "error": str(e)}

    if action_norm in {"stop", "shutdown"}:
        try:
            client = _get_external_client_if_configured()
            if client is None:
                return {
                    "success": False,
                    "action": "stop",
                    "error": "Attach mode not configured (ROBOTMCP_ATTACH_HOST not set)",
                }
            resp = client.stop()
            return {
                "success": bool(resp.get("success")),
                "action": "stop",
                "response": resp,
            }
        except Exception as e:
            logger.error(f"attach stop failed: {e}")
            return {"success": False, "action": "stop", "error": str(e)}

    # Phase 2a ADR-004: cleanup action - clean expired sessions and check bridge health
    if action_norm in {"cleanup", "clean"}:
        try:
            # Clean expired local sessions
            expired_count = execution_engine.session_manager.cleanup_expired_sessions()

            # Check bridge health if configured
            client = _get_external_client_if_configured()
            bridge_status: Dict[str, Any] = {"configured": False}

            if client is not None:
                try:
                    health = _check_bridge_health(client)
                    bridge_status = {
                        "configured": True,
                        "reachable": health.get("healthy", False),
                        "context_active": health.get("context_active", False),
                    }
                    if not health.get("healthy"):
                        bridge_status["error"] = health.get("error")
                        bridge_status["recovery_hint"] = health.get("recovery_hint")
                except Exception as health_err:
                    bridge_status = {
                        "configured": True,
                        "reachable": False,
                        "error": str(health_err),
                    }

            return {
                "success": True,
                "action": "cleanup",
                "sessions_cleaned": expired_count,
                "remaining_sessions": len(execution_engine.session_manager.sessions),
                "bridge_status": bridge_status,
            }
        except Exception as e:
            logger.error(f"attach cleanup failed: {e}")
            return {"success": False, "action": "cleanup", "error": str(e)}

    # Phase 2a ADR-004: reset action - stop bridge and clean all local sessions
    if action_norm in {"reset", "reconnect"}:
        try:
            client = _get_external_client_if_configured()
            if client is None:
                return {
                    "success": False,
                    "action": "reset",
                    "error": "Attach mode not configured (ROBOTMCP_ATTACH_HOST not set)",
                }

            # Stop current bridge (sets stop flag)
            stop_resp = client.stop()

            # Clean all local sessions
            sessions_cleaned = execution_engine.session_manager.cleanup_all_sessions()

            return {
                "success": True,
                "action": "reset",
                "bridge_stopped": bool(stop_resp.get("success")),
                "sessions_cleaned": sessions_cleaned,
                "recovery_hint": "RF process must call 'MCP Serve' again to restart bridge",
            }
        except Exception as e:
            logger.error(f"attach reset failed: {e}")
            return {"success": False, "action": "reset", "error": str(e)}

    # Phase 2a ADR-004: disconnect_all action - force stop bridge and terminate all
    if action_norm in {"disconnect_all", "terminate", "force_stop"}:
        try:
            client = _get_external_client_if_configured()
            if client is None:
                return {
                    "success": False,
                    "action": "disconnect_all",
                    "error": "Attach mode not configured (ROBOTMCP_ATTACH_HOST not set)",
                }

            # Try force stop (new verb) with fallback to regular stop
            try:
                stop_resp = client.force_stop()
            except Exception:
                # Fall back to regular stop if force_stop not implemented
                stop_resp = client.stop()

            # Clean all local sessions
            sessions_cleaned = execution_engine.session_manager.cleanup_all_sessions()

            return {
                "success": True,
                "action": "disconnect_all",
                "bridge_stopped": bool(stop_resp.get("success")),
                "sessions_cleaned": sessions_cleaned,
                "hint": "All connections terminated. Bridge must be restarted via MCP Serve.",
            }
        except Exception as e:
            logger.error(f"attach disconnect_all failed: {e}")
            return {"success": False, "action": "disconnect_all", "error": str(e)}

    return {
        "success": False,
        "error": f"Unsupported action '{action}'",
        "action": action,
    }


@mcp.tool(
    name="attach_status",
    description="Report attach-mode configuration and bridge health. Indicates whether execute_step(use_context=true) will route externally.",
    enabled=False,
)
async def attach_status() -> Dict[str, Any]:
    try:
        client = _get_external_client_if_configured()
        configured = client is not None
        mode = os.environ.get("ROBOTMCP_ATTACH_DEFAULT", "auto").strip().lower()
        strict = os.environ.get("ROBOTMCP_ATTACH_STRICT", "0").strip() in {
            "1",
            "true",
            "yes",
        }
        if not configured:
            return {
                "configured": False,
                "default_mode": mode,
                "strict": strict,
                "hint": "Set ROBOTMCP_ATTACH_HOST to enable attach mode.",
            }
        diag = client.diagnostics()
        return {
            "configured": True,
            "host": client.host,
            "port": client.port,
            "reachable": bool(diag.get("success")),
            "diagnostics": diag.get("result"),
            "default_mode": mode,
            "strict": strict,
            "hint": "execute_step(..., use_context=true) routes to the bridge when reachable.",
        }
    except Exception as e:
        logger.error(f"attach_status failed: {e}")
        return {"configured": False, "error": str(e)}


@mcp.tool(
    name="attach_stop_bridge",
    description="Send a stop command to the external attach bridge (McpAttach) to exit MCP Serve in the debugged suite.",
    enabled=False,
)
async def attach_stop_bridge() -> Dict[str, Any]:
    try:
        client = _get_external_client_if_configured()
        if client is None:
            return {
                "success": False,
                "error": "Attach mode not configured (ROBOTMCP_ATTACH_HOST not set)",
            }
        resp = client.stop()
        ok = bool(resp.get("success"))
        return {"success": ok, "response": resp}
    except Exception as e:
        logger.error(f"attach_stop_bridge failed: {e}")
        return {"success": False, "error": str(e)}


# note: variable tools consolidated into get_context_variables/set_variables with attach routing


@mcp.tool(
    name="import_resource",
    description="Import a Robot Framework resource file into the session RF Namespace.",
    enabled=False,
)
async def import_resource(session_id: str, path: str) -> Dict[str, Any]:
    def _local_call() -> Dict[str, Any]:
        mgr = get_rf_native_context_manager()
        return mgr.import_resource_for_session(session_id, path)

    def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
        return client.import_resource(path)

    return _call_attach_tool_with_fallback(
        "import_resource", _external_call, _local_call
    )


@mcp.tool(
    name="import_custom_library",
    description="Import a custom Robot Framework library (module name or file path) into the session RF Namespace.",
    enabled=False,
)
async def import_custom_library(
    session_id: str,
    name_or_path: str,
    args: List[str] | None = None,
    alias: str | None = None,
) -> Dict[str, Any]:
    def _local_call() -> Dict[str, Any]:
        mgr = get_rf_native_context_manager()
        return mgr.import_library_for_session(
            session_id, name_or_path, tuple(args or ()), alias
        )

    def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
        return client.import_library(name_or_path, list(args or ()), alias)

    return _call_attach_tool_with_fallback(
        "import_custom_library", _external_call, _local_call
    )


@mcp.tool(
    name="list_available_keywords",
    description="List available keywords from imported libraries and resources in the session RF Namespace.",
    enabled=False,
)
async def list_available_keywords(session_id: str) -> Dict[str, Any]:
    def _local_call() -> Dict[str, Any]:
        mgr = get_rf_native_context_manager()
        return mgr.list_available_keywords(session_id)

    def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
        r = client.list_keywords()
        return {
            "success": r.get("success", False),
            "session_id": session_id,
            "external": True,
            "keywords_by_library": r.get("result"),
        }

    return _call_attach_tool_with_fallback(
        "list_available_keywords", _external_call, _local_call
    )


async def _get_session_keyword_documentation_payload(
    session_id: str, keyword_name: str
) -> Dict[str, Any]:
    def _local_call() -> Dict[str, Any]:
        mgr = get_rf_native_context_manager()
        return mgr.get_keyword_documentation(session_id, keyword_name)

    def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
        r = client.get_keyword_doc(keyword_name)
        if r.get("success"):
            return {
                "success": True,
                "session_id": session_id,
                "name": r["result"]["name"],
                "source": r["result"]["source"],
                "doc": r["result"]["doc"],
                "args": r["result"].get("args", []),
                "type": "external",
            }
        return {
            "success": False,
            "error": r.get("error", "failed"),
            "session_id": session_id,
        }

    return _call_attach_tool_with_fallback(
        "get_session_keyword_documentation", _external_call, _local_call
    )


@mcp.tool(
    name="get_session_keyword_documentation",
    description="Get documentation for a keyword (library or resource) available in the session RF Namespace.",
    enabled=False,
)
async def get_session_keyword_documentation(
    session_id: str, keyword_name: str
) -> Dict[str, Any]:
    return await _get_session_keyword_documentation_payload(session_id, keyword_name)


class BridgeHealthMonitor:
    """Monitors bridge health with periodic heartbeats.

    This class implements Phase 4 of ADR-004: Optional Heartbeat Monitor.
    It runs a background asyncio task that periodically checks the health
    of the debug bridge connection and triggers session cleanup after
    consecutive failures exceed a threshold.

    Configuration via environment variables:
        ROBOTMCP_BRIDGE_HEARTBEAT: Enable/disable (0 or 1, default 0)
        ROBOTMCP_HEARTBEAT_INTERVAL: Check interval in seconds (default 60)
        ROBOTMCP_HEARTBEAT_THRESHOLD: Failures before cleanup (default 3)
    """

    def __init__(self, interval_seconds: int = 60, failure_threshold: int = 3):
        """Initialize the health monitor.

        Args:
            interval_seconds: Time between health checks in seconds.
            failure_threshold: Number of consecutive failures before triggering cleanup.
        """
        self.interval = interval_seconds
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.last_healthy: Optional[float] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the health monitor loop."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Bridge health monitor started (interval: {self.interval}s)")

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Bridge health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop that runs health checks at the configured interval."""
        while self._running:
            await asyncio.sleep(self.interval)
            await self._check_health()

    async def _check_health(self) -> None:
        """Perform a single health check against the bridge.

        Updates consecutive_failures counter and triggers cleanup if threshold exceeded.
        """
        client = _get_external_client_if_configured()
        if client is None:
            # No attach mode configured, nothing to monitor
            return

        try:
            health = _check_bridge_health(client)

            if health.get("healthy"):
                self.last_healthy = time.time()
                self.consecutive_failures = 0
                logger.debug(
                    f"Bridge health check passed (context_active: {health.get('context_active')})"
                )
            else:
                self.consecutive_failures += 1
                logger.warning(
                    f"Bridge health check failed ({self.consecutive_failures}/{self.failure_threshold}): "
                    f"{health.get('recovery_hint', 'unknown issue')}"
                )

                if self.consecutive_failures >= self.failure_threshold:
                    await self._handle_unhealthy_bridge()
        except Exception as e:
            self.consecutive_failures += 1
            logger.warning(f"Bridge health check error: {e}")

            if self.consecutive_failures >= self.failure_threshold:
                await self._handle_unhealthy_bridge()

    async def _handle_unhealthy_bridge(self) -> None:
        """Handle persistent bridge failure by cleaning expired sessions."""
        logger.warning("Bridge unhealthy for extended period, cleaning stale sessions")
        try:
            cleaned = execution_engine.session_manager.cleanup_expired_sessions()
            if cleaned > 0:
                logger.info(f"Cleaned {cleaned} expired sessions due to unhealthy bridge")
            # Reset counter after cleanup to avoid repeated cleanups
            self.consecutive_failures = 0
        except Exception as e:
            logger.error(f"Failed to cleanup sessions after unhealthy bridge detection: {e}")


# Global health monitor instance (set in main() if enabled)
_health_monitor: Optional[BridgeHealthMonitor] = None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RobotMCP server entry point with optional Django frontend."
    )
    parser.add_argument(
        "--with-frontend",
        dest="frontend_enabled_flag",
        action="store_const",
        const=True,
        help="Start the optional Django-based frontend alongside the MCP server.",
    )
    parser.add_argument(
        "--without-frontend",
        dest="frontend_enabled_flag",
        action="store_const",
        const=False,
        help="Disable the optional frontend even if the environment enables it.",
    )
    parser.add_argument(
        "--frontend-host",
        dest="frontend_host",
        help="Host interface for the frontend server (default 127.0.0.1).",
    )
    parser.add_argument(
        "--frontend-port",
        dest="frontend_port",
        type=int,
        help="Port for the frontend server (default 8001).",
    )
    parser.add_argument(
        "--frontend-base-path",
        dest="frontend_base_path",
        help="Base path prefix for the frontend (default '/').",
    )
    parser.add_argument(
        "--frontend-debug",
        dest="frontend_debug",
        action="store_const",
        const=True,
        help="Enable Django debug mode for the frontend.",
    )
    parser.add_argument(
        "--frontend-no-debug",
        dest="frontend_debug",
        action="store_const",
        const=False,
        help="Disable Django debug mode for the frontend.",
    )
    parser.add_argument(
        "--transport",
        dest="transport",
        choices=["stdio", "http", "sse"],
        help="Transport to use for the MCP server (default: stdio).",
    )
    parser.add_argument(
        "--host",
        dest="host",
        help="Host/interface for HTTP transport (default 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        help="Port for HTTP transport (default 8000).",
    )
    parser.add_argument(
        "--path",
        dest="path",
        help="Path for HTTP/streamable endpoints (default '/').",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        help="Log level for the MCP server (e.g., INFO, DEBUG).",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Start the RobotMCP server, optionally booting the Django frontend."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    try:
        _log_attach_banner()
    except Exception:
        pass

    # ADR-004 Phase 1: Startup validation and cleanup
    try:
        _startup_bridge_validation()
    except Exception:
        pass

    from robotmcp.frontend.config import (
        FrontendConfig,
        build_frontend_config,
        frontend_enabled_from_env,
    )

    enable_frontend = frontend_enabled_from_env(default=False)
    if args.frontend_enabled_flag is not None:
        enable_frontend = args.frontend_enabled_flag

    frontend_config = FrontendConfig(enabled=False)
    if enable_frontend:
        if not _frontend_dependencies_available():
            logger.warning(
                "Frontend requested but Django/uvicorn dependencies are missing. "
                "Install with 'pip install rf-mcp[frontend]' or disable the frontend."
            )
            enable_frontend = False
        else:
            frontend_config = build_frontend_config(
                enabled=True,
                host=args.frontend_host,
                port=args.frontend_port,
                base_path=args.frontend_base_path,
                debug=args.frontend_debug,
            )
            _install_frontend_lifespan(frontend_config)

    if enable_frontend:
        logger.info("Starting RobotMCP with frontend at %s", frontend_config.url)
    else:
        logger.info("Starting RobotMCP without frontend")

    # Configure bridge health monitor if enabled (Phase 4 of ADR-004)
    # Environment variables:
    #   ROBOTMCP_BRIDGE_HEARTBEAT: 0 or 1 (default 0 - disabled)
    #   ROBOTMCP_HEARTBEAT_INTERVAL: seconds (default 60)
    #   ROBOTMCP_HEARTBEAT_THRESHOLD: failures before cleanup (default 3)
    heartbeat_enabled = os.environ.get("ROBOTMCP_BRIDGE_HEARTBEAT", "0").strip() in {
        "1",
        "true",
        "yes",
    }
    if heartbeat_enabled:
        # Only enable if attach mode is configured
        if _get_external_client_if_configured() is not None:
            try:
                heartbeat_interval = int(
                    os.environ.get("ROBOTMCP_HEARTBEAT_INTERVAL", "60")
                )
                heartbeat_threshold = int(
                    os.environ.get("ROBOTMCP_HEARTBEAT_THRESHOLD", "3")
                )
                _install_health_monitor_lifespan(
                    interval_seconds=heartbeat_interval,
                    failure_threshold=heartbeat_threshold,
                )
                logger.info(
                    f"Bridge health monitor enabled (interval: {heartbeat_interval}s, "
                    f"threshold: {heartbeat_threshold} failures)"
                )
            except ValueError as e:
                logger.warning(f"Invalid heartbeat configuration, monitor disabled: {e}")
        else:
            logger.debug(
                "Bridge health monitor requested but attach mode not configured, skipping"
            )

    try:
        run_kwargs = {}

        # Default to stdio when no transport is provided to remain backward compatible
        transport = args.transport or "stdio"
        run_kwargs["transport"] = transport

        # log_level is accepted by both stdio and http
        if args.log_level:
            run_kwargs["log_level"] = args.log_level

        # Only pass host/port/path when using HTTP/SSE transports
        if transport != "stdio":
            if args.host:
                run_kwargs["host"] = args.host
            if args.port:
                run_kwargs["port"] = args.port
            if args.path:
                run_kwargs["path"] = args.path

        mcp.run(**run_kwargs)
    except KeyboardInterrupt:
        logger.info("RobotMCP interrupted by user")
    finally:
        # Shutdown instruction learning and persist metrics
        try:
            hooks = _get_instruction_hooks()
            shutdown_result = hooks.shutdown()
            if shutdown_result.get("persisted"):
                logger.info(
                    f"Instruction learning data persisted: "
                    f"{shutdown_result.get('sessions_ended', 0)} sessions saved"
                )
        except Exception:
            logger.debug("Failed to shutdown instruction learning", exc_info=True)

        try:
            execution_engine.session_manager.cleanup_all_sessions()
        except Exception:
            logger.debug("Failed to cleanup sessions on shutdown", exc_info=True)

        if _frontend_controller:
            try:
                asyncio.run(_frontend_controller.stop())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_frontend_controller.stop())
                loop.close()


if __name__ == "__main__":
    main()
