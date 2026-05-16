"""N1: Tests verifying that intent_action click+force=True dispatches to
Click With Options (not Click) for Browser Library sessions.

Browser's Click(selector, button="left") does not accept force= as a named
arg; Click With Options(selector, *clickOptions) is the correct escape hatch.
"""
from __future__ import annotations

__test__ = True

from typing import Any, Dict, List, Optional

import pytest

from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import IntentTarget, NormalizedLocator


class _SessionLookup:
    def __init__(self, library: str = "Browser") -> None:
        self._lib = library

    def get_active_web_library(self, session_id: str) -> Optional[str]:
        return self._lib

    def get_imported_libraries(self, session_id: str) -> List[str]:
        return [self._lib]

    def get_platform_type(self, session_id: str) -> str:
        return "web"


class _PassthroughNormalizer:
    def normalize(self, target: IntentTarget, target_library: str) -> NormalizedLocator:
        return NormalizedLocator(
            value=target.locator,
            source_locator=target.locator,
            target_library=target_library,
            strategy_applied="pass_through",
            was_transformed=False,
        )


def _adapter(library: str = "Browser") -> IntentActionAdapter:
    registry = IntentRegistry.with_builtins()
    resolver = IntentResolver(
        registry=registry,
        session_lookup=_SessionLookup(library),
        normalizer=_PassthroughNormalizer(),
    )
    return IntentActionAdapter(resolver=resolver)


def _apply_force_n1(resolution: Dict[str, Any], force: bool) -> tuple[str, list[str]]:
    """Replicate the N1 force-application logic from server.py for unit testing."""
    lib = resolution.get("library", "")
    resolved_keyword = resolution["keyword"]
    resolved_args = list(resolution["arguments"])

    if force and lib == "Browser":
        force_keyword = resolution.get("force_keyword")
        if force_keyword:
            resolved_keyword = force_keyword
        if "force=True" not in resolved_args:
            resolved_args.append("force=True")

    return resolved_keyword, resolved_args


# ---------------------------------------------------------------------------
# CLICK + force=True -> Click With Options
# ---------------------------------------------------------------------------

def test_click_browser_force_true_routes_to_click_with_options():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=foo")
    keyword, args = _apply_force_n1(resolution, force=True)
    assert keyword == "Click With Options", (
        f"Expected 'Click With Options', got '{keyword}'"
    )
    assert "force=True" in args


def test_click_browser_force_true_has_selector_as_first_arg():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=foo")
    keyword, args = _apply_force_n1(resolution, force=True)
    assert args[0] == "id=foo"
    assert "force=True" in args


def test_click_browser_no_force_stays_click():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=foo")
    keyword, args = _apply_force_n1(resolution, force=False)
    assert keyword == "Click"
    assert "force=True" not in args


def test_resolution_exposes_force_keyword_for_browser_click():
    """Adapter must include force_keyword='Click With Options' in returned dict."""
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=foo")
    assert resolution.get("force_keyword") == "Click With Options"


# ---------------------------------------------------------------------------
# FILL TEXT + force=True -> stays Fill Text (not a click-family keyword)
# ---------------------------------------------------------------------------

def test_fill_browser_force_true_stays_fill_text():
    """Fill Text does not have a force_keyword; keyword must remain Fill Text."""
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="fill", target="id=name", value="Alice")
    keyword, args = _apply_force_n1(resolution, force=True)
    assert keyword == "Fill Text"
    assert "force=True" in args


def test_fill_browser_force_keyword_is_none():
    """Fill Text mapping should not declare a force_keyword."""
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="fill", target="id=name", value="Alice")
    assert resolution.get("force_keyword") is None


# ---------------------------------------------------------------------------
# SeleniumLibrary click: no force routing (SeleniumLibrary has no force= support)
# ---------------------------------------------------------------------------

def test_click_selenium_force_true_not_swapped():
    adapter = _adapter("SeleniumLibrary")
    resolution = adapter.resolve_intent(intent="click", target="id=foo")
    keyword, args = _apply_force_n1(resolution, force=True)
    # SeleniumLibrary block not entered; keyword unchanged, no force=True appended
    assert "force=True" not in args
    assert keyword != "Click With Options"


# ---------------------------------------------------------------------------
# Idempotency: force=True not duplicated
# ---------------------------------------------------------------------------

def test_force_true_not_duplicated():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=foo")
    resolution = dict(resolution)
    resolution["arguments"] = list(resolution["arguments"]) + ["force=True"]
    keyword, args = _apply_force_n1(resolution, force=True)
    assert args.count("force=True") == 1
