"""Main MCP Server implementation for Robot Framework integration."""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from fastmcp import FastMCP

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
from robotmcp.models.session_models import PlatformType
from robotmcp.plugins import get_library_plugin_manager
from robotmcp.utils.server_integration import initialize_enhanced_serialization

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from robotmcp.frontend.controller import FrontendServerController


# Initialize FastMCP server
mcp = FastMCP("Robot Framework MCP Server")

# Optional reference to the running frontend controller
_frontend_controller: "FrontendServerController | None" = None


def _get_external_client_if_configured() -> ExternalRFClient | None:
    """Return an ExternalRFClient when attach mode is configured via env.

    Env vars:
    - ROBOTMCP_ATTACH_HOST (required to enable attach mode)
    - ROBOTMCP_ATTACH_PORT (optional, defaults 7317)
    - ROBOTMCP_ATTACH_TOKEN (optional, defaults 'change-me')
    """
    try:
        host = os.environ.get("ROBOTMCP_ATTACH_HOST")
        if not host:
            return None
        port = int(os.environ.get("ROBOTMCP_ATTACH_PORT", "7317"))
        token = os.environ.get("ROBOTMCP_ATTACH_TOKEN", "change-me")
        return ExternalRFClient(host=host, port=port, token=token)
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
    """Consolidated plugin management entry point.

    Args:
        action: 'list', 'reload', or 'diagnose'.
        plugin_name: Required when using action='diagnose'.
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
) -> Dict[str, Any]:
    """Recommend libraries, generate sampling prompts, or merge sampled payloads."""

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

    rec = library_recommender.recommend_libraries(
        scenario, context=context, max_recommendations=max_recommendations
    )
    if not rec.get("success"):
        return {"success": False, "error": rec.get("error", "Recommendation failed")}

    recommendations = rec.get("recommendations", [])
    recommended_names = [
        r.get("library_name") for r in recommendations if r.get("library_name")
    ]

    result: Dict[str, Any] = {
        "success": True,
        "mode": "direct",
        "scenario": scenario,
        "context": context,
        "recommended_libraries": recommended_names,
        "recommendations": recommendations,
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

    return result


@mcp.tool
async def analyze_scenario(
    scenario: str, context: str = "web", session_id: str = None
) -> Dict[str, Any]:
    """Process natural language test description into structured test intent.

    CRITICAL: This tool ALWAYS creates a session for your test execution.
    Use the returned session_id in ALL subsequent tool calls (execute_step, build_test_suite, etc.)

    RECOMMENDED WORKFLOW - STEP 1 OF 4:
    This tool should be used as the FIRST step in the Robot Framework automation workflow:
    1. ✅ analyze_scenario (THIS TOOL) - Creates session and understands requirements
    2. ➡️ recommend_libraries - Get targeted library suggestions
    3. ➡️ execute_step - Execute steps using the SAME session_id
    4. ➡️ build_test_suite - Build suite using the SAME session_id

    Using this order prevents unnecessary library checks and pip installations by ensuring
    you only verify libraries that are actually relevant to the user's scenario.

    NEW FEATURE: Automatic Session Creation
    - If session_id not provided: Creates new unique session ID
    - If session_id provided: Uses existing session or creates new one
    - Session is auto-configured based on scenario analysis and explicit library preferences
    - Returns session_id that MUST be used in all subsequent calls

    Args:
        scenario: Human language scenario description
        context: Optional context about the application (web, mobile, API, etc.)
        session_id: Optional session ID to create and auto-configure for this scenario

    Returns:
        Structured test intent with session_info containing session_id for subsequent calls.
        Session is automatically configured with optimal library choices for the scenario.
    """
    # Analyze the scenario first
    result = await nlp_processor.analyze_scenario(scenario, context)

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
    }
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

    return result


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
    """Unified keyword discovery across semantic, pattern, catalog, and session scopes."""

    strategy_norm = (strategy or "semantic").strip().lower()
    current_state = current_state or {}
    limit_value: int | None = None
    if limit is not None:
        try:
            limit_value = int(limit)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            limit_value = None

    if strategy_norm in {"semantic", "intent"}:
        discovery = await keyword_matcher.discover_keywords(
            query, context, current_state
        )
        return {
            "success": bool(discovery.get("success", True)),
            "strategy": "semantic",
            "query": query,
            "result": discovery,
        }

    if strategy_norm in {"pattern", "search"}:
        await _ensure_all_session_libraries_loaded()
        matches = execution_engine.search_keywords(query)
        if limit_value is not None:
            matches = matches[:limit_value]
        return {
            "success": True,
            "strategy": "pattern",
            "query": query,
            "results": matches,
        }

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
        if limit_value is not None:
            catalog = catalog[:limit_value]
        return {
            "success": True,
            "strategy": "catalog",
            "query": query,
            "library": library_name,
            "results": catalog,
        }

    if strategy_norm in {"session", "namespace"}:
        if not session_id:
            return {
                "success": False,
                "error": "session_id is required when strategy='session'",
            }
        mgr = get_rf_native_context_manager()
        payload = mgr.list_available_keywords(session_id)
        payload.update({"strategy": "session", "query": query})
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
    scope: str = "test",
) -> Dict[str, Any]:
    """Consolidated session management (initialize, import, set variables)."""

    action_norm = (action or "").strip().lower()
    session = execution_engine.session_manager.get_or_create_session(session_id)

    if action_norm in {"init", "initialize", "bootstrap"}:
        loaded: List[str] = []
        problems: List[Dict[str, Any]] = []

        if libraries:
            for library in libraries:
                try:
                    session.import_library(library)
                    session.loaded_libraries.add(library)
                    loaded.append(library)
                except Exception as lib_error:
                    problems.append({"library": library, "error": str(lib_error)})

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
            for name, value in iterable:
                key = name if name.startswith("${") else f"${{{name}}}"
                session.set_variable(key, value)
                set_vars.append(name)

        return {
            "success": True,
            "action": "init",
            "session_id": session_id,
            "libraries_loaded": list(session.loaded_libraries),
            "variables_set": set_vars,
            "import_issues": problems,
            "note": "Context mode is managed via session namespace; use execute_step(use_context=True) when needed.",
        }

    if action_norm in {"import_resource", "resource"}:
        if not resource_path:
            return {"success": False, "error": "resource_path is required"}

        def _local_call() -> Dict[str, Any]:
            mgr = get_rf_native_context_manager()
            return mgr.import_resource_for_session(session_id, resource_path)

        def _external_call(client: ExternalRFClient) -> Dict[str, Any]:
            return client.import_resource(resource_path)

        result = _call_attach_tool_with_fallback(
            "import_resource", _external_call, _local_call
        )
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

        set_kw = {
            "test": "Set Test Variable",
            "suite": "Set Suite Variable",
            "global": "Set Global Variable",
        }.get(scope.lower(), "Set Test Variable")

        results: Dict[str, bool] = {}
        client = _get_external_client_if_configured()
        if client is not None:
            for name, value in data.items():
                try:
                    resp = client.set_variable(name, value)
                    results[name] = bool(resp.get("success"))
                except Exception:
                    results[name] = False
            return {
                "success": all(results.values()),
                "action": "set_variables",
                "session_id": session_id,
                "set": list(results.keys()),
                "scope": scope,
                "external": True,
            }

        for name, value in data.items():
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
            "action": "set_variables",
            "session_id": session_id,
            "set": list(results.keys()),
            "scope": scope,
        }

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
    """Unified flow control wrapper over if/for/try structures."""

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

    return payload


@mcp.tool
async def execute_step(
    keyword: str,
    arguments: List[str] = None,
    session_id: str = "default",
    raise_on_failure: bool = True,
    detail_level: str = "minimal",
    scenario_hint: str = None,
    assign_to: Union[str, List[str]] = None,
    use_context: bool | None = None,
    mode: str = "keyword",
    expression: str | None = None,
) -> Dict[str, Any]:
    """Execute a single test step using Robot Framework API.

    STEPWISE TEST DEVELOPMENT GUIDANCE:
    - ALWAYS execute and verify each keyword individually BEFORE building test suites
    - Test each step to confirm it works as expected
    - Only add verified keywords to .robot files
    - Use this method to validate arguments and behavior step-by-step
    - Build incrementally: execute_step() → verify → add to suite

    Args:
        keyword: Robot Framework keyword name
        arguments: Arguments for the keyword (supports both positional and named: ["arg1", "param=value"])
        session_id: Session identifier for maintaining context
        raise_on_failure: If True, raises exception for failed steps (proper MCP failure reporting).
                         If False, returns failure details in response (for debugging/analysis).
        detail_level: Level of detail in response ('minimal', 'standard', 'full').
                     'minimal' reduces response size for AI agents by ~80-90%.
        scenario_hint: Optional scenario text for intelligent library auto-configuration.
                      When provided on first call, automatically configures the session
                      based on detected scenario type and explicit library preferences.
        assign_to: Variable name(s) to assign the keyword's return value to.
                  Single string for single assignment: "result" creates ${result}
                  List of strings for multi-assignment: ["first", "rest"] creates ${first}, ${rest}
        use_context: If True, executes within full RF context (maintains variables, state across calls).
                    This enables proper variable scoping, built-in keyword functionality, and
                    library state persistence.
        mode: 'keyword' (default) runs the provided keyword name; 'evaluate' executes a BuiltIn.Evaluate call.
        expression: Optional expression used when mode='evaluate'. Falls back to keyword or first argument.
    """
    arguments = list(arguments or [])
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
            logger.error(f"ATTACH mode error: {err}")
            if strict or mode == "force":
                raise Exception(
                    f"Attach bridge call failed: {err}. Is MCP Serve running and token/port correct?"
                )
            # Fallback to local execution
            logger.warning("ATTACH unreachable; falling back to local execution")
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

    # Local execution path
    result = await execution_engine.execute_step(
        keyword_to_run,
        arguments,
        session_id,
        detail_level,
        scenario_hint=scenario_hint,
        assign_to=assign_to,
        use_context=bool(use_context),
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
    return result


async def _get_application_state_payload(
    state_type: str = "all",
    elements_of_interest: List[str] | None = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    if elements_of_interest is None:
        elements_of_interest = []
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
    """Generate Robot Framework test suite from successful steps with intelligent session resolution.

    IMPORTANT: Only use AFTER validating all steps individually with execute_step().
    This tool generates .robot files from previously executed and verified steps.
    Do NOT write test suites before confirming each keyword works correctly.

    Enhanced Session Resolution:
    - If session_id provided and valid: Uses that session
    - If session_id empty/invalid: Automatically finds most suitable session with steps
    - Provides clear guidance on session issues and recovery options

    Recommended workflow:
    1. Use analyze_scenario() to create configured session
    2. Use execute_step() to test each keyword with the SAME session_id
    3. Use build_test_suite() with the SAME session_id to create .robot files

    Args:
        test_name: Name for the test case
        session_id: Session with executed steps (auto-resolves if empty/invalid)
        tags: Test tags
        documentation: Test documentation
        remove_library_prefixes: Remove library prefixes from keywords (default: True)
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
    # Bridge path: try Browser.Get Page Source or SeleniumLibrary.Get Source in live debug session
    client = _get_external_client_if_configured()
    if client is not None:
        last_error = None
        # Prefer Browser's keyword
        for kw in ("Get Page Source", "Get Source"):
            resp = client.run_keyword(kw, [])
            if resp.get("success") and resp.get("result") is not None:
                src = resp.get("result")
                # Minimal normalized payload with external flag
                aria_snapshot_payload = {
                    "success": False,
                    "selector": "css=html",
                    "library": "Browser",
                }
                if include_reduced_dom:
                    aria_snapshot_payload["error"] = "not_supported_in_attach_mode"
                else:
                    aria_snapshot_payload["skipped"] = True

                return {
                    "success": True,
                    "external": True,
                    "keyword_used": kw,
                    "page_source": src,
                    "metadata": {"full": True, "filtered": False},
                    "aria_snapshot": aria_snapshot_payload,
                }
            last_error = resp.get("error") or resp.get("message")
        logger.warning(
            "Attach bridge could not retrieve page source (last error: %s); falling back to local execution",
            last_error,
        )

    # Local path
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
    """Check if Robot Framework libraries are available before installation.

    RECOMMENDED WORKFLOW - STEP 3 OF 3:
    This tool should be used as the THIRD step in the Robot Framework automation workflow:
    1. ✅ analyze_scenario - Understand what the user wants to accomplish
    2. ✅ recommend_libraries - Get targeted library suggestions for the scenario
    3. ✅ check_library_availability (THIS TOOL) - Verify only the recommended libraries

    CRITICAL: Do NOT call this tool first! It may return empty results if called before
    the Robot Framework environment is initialized, leading to unnecessary pip installations.

    PREFERRED INPUT: Use the library recommendations from recommend_libraries as the
    'libraries' parameter to avoid checking irrelevant libraries.

    FALLBACK INITIALIZATION: If you must call this tool without the recommended workflow,
    first call 'get_available_keywords' or 'execute_step' to initialize library discovery,
    then re-run this check.

    Args:
        libraries: List of library names to check (preferably from recommend_libraries output)

    Returns:
        Dict with availability status, installation suggestions, and workflow guidance.
        Includes smart hints if called in wrong order or without initialization.

    Example workflow:
        scenario_result = await analyze_scenario("I want to test a web form")
        recommendations = await recommend_libraries(scenario_result["scenario"])
        availability = await check_library_availability(recommendations["recommended_libraries"])
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
                resp = client.set_variable(name, value)
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
    """Retrieve keyword/library documentation or parse signatures from a single entry point."""

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
    """Set explicit library search order for keyword resolution (like RF Set Library Search Order).

    This tool implements Robot Framework's Set Library Search Order concept, allowing explicit
    control over which library's keywords take precedence when multiple libraries have the
    same keyword name.

    Args:
        libraries: List of library names in priority order (highest priority first)
        session_id: Session identifier to configure

    Returns:
        Dict with success status, applied search order, and any warnings about invalid libraries

    Example:
        # Prioritize SeleniumLibrary over Browser Library for web automation
        await set_library_search_order(["SeleniumLibrary", "BuiltIn", "Collections"], "web_session")

        # Prioritize RequestsLibrary for API testing
        await set_library_search_order(["RequestsLibrary", "BuiltIn", "String"], "api_session")
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
    try:
        # Helper to sanitize values: return scalars as-is; for complex objects, return their type name.
        def _sanitize(val: Any) -> Any:
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            # Avoid serializing complex/large objects
            return f"<{type(val).__name__}>"

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
        session = execution_engine.session_manager.get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found",
                "session_id": session_id,
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
    try:
        session = execution_engine.session_manager.get_session(session_id)

        if not session:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found",
                "available_sessions": execution_engine.session_manager.get_all_session_ids(),
            }

        return {"success": True, "session_info": session.get_session_info()}

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
    """Provide locator/selector guidance for Browser/Selenium/Appium libraries."""

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
    """Execute or validate Robot Framework suites depending on `mode`.

    mode='dry' / 'validate' performs a dry run; mode='full' executes the suite.
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
    """Inspect or control attach bridge configuration."""

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

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("RobotMCP interrupted by user")
    finally:
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
