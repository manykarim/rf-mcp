"""Tests for P6: force= parameter on intent_action.

Verifies that force=True appends "force=True" to Browser Click and Fill Text.
"""
from __future__ import annotations

__test__ = True

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import IntentTarget, NormalizedLocator


class _SessionLookup:
    def __init__(self, library: str = "Browser"):
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


# We test the resolver output (keyword + args), then test the server-level
# force handling via a small async helper that mimics what server.py does.


def _server_apply_force(resolution: Dict[str, Any], force: bool, lib: str) -> tuple:
    """Replicate the N1 server.py force-application logic for testing."""
    resolved_keyword = resolution["keyword"]
    resolved_args = list(resolution["arguments"])

    if force and lib == "Browser":
        # N1: swap to force_keyword when the mapping declares one
        force_keyword = resolution.get("force_keyword")
        if force_keyword:
            resolved_keyword = force_keyword
        if "force=True" not in resolved_args:
            resolved_args.append("force=True")

    return resolved_keyword, resolved_args


def test_click_browser_force_true_appends_force_arg():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(
        intent="click",
        target="id=submit",
    )
    keyword, args = _server_apply_force(resolution, force=True, lib="Browser")
    # N1: Click swaps to Click With Options when force=True for Browser
    assert keyword == "Click With Options"
    assert "force=True" in args


def test_click_browser_force_false_no_force_arg():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(
        intent="click",
        target="id=submit",
    )
    keyword, args = _server_apply_force(resolution, force=False, lib="Browser")
    assert "force=True" not in args


def test_fill_browser_force_true_appends_force_arg():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(
        intent="fill",
        target="id=name",
        value="Alice",
    )
    keyword, args = _server_apply_force(resolution, force=True, lib="Browser")
    assert keyword == "Fill Text"
    assert "force=True" in args


def test_fill_browser_force_false_no_force_arg():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(
        intent="fill",
        target="id=name",
        value="Alice",
    )
    keyword, args = _server_apply_force(resolution, force=False, lib="Browser")
    assert "force=True" not in args


def test_force_true_not_duplicated_when_already_present():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(
        intent="click",
        target="id=submit",
    )
    # Pre-add force=True (simulate double-call)
    resolution = dict(resolution)
    resolution["arguments"] = list(resolution["arguments"]) + ["force=True"]
    keyword, args = _server_apply_force(resolution, force=True, lib="Browser")
    assert args.count("force=True") == 1


def test_force_selenium_does_not_append_force_arg():
    adapter = _adapter("SeleniumLibrary")
    resolution = adapter.resolve_intent(
        intent="click",
        target="id=submit",
    )
    keyword, args = _server_apply_force(resolution, force=True, lib="SeleniumLibrary")
    # force is Browser-only — SeleniumLibrary doesn't support force= arg
    assert "force=True" not in args
