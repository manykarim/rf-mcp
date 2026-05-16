"""Tests for P6: timeout= parameter on intent_action.

Verifies that timeout is passed into options["timeout"] for transformers and
appended as "timeout=<value>" for Browser Library.
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


def _server_apply_timeout(
    resolution: Dict[str, Any], timeout: Optional[str], lib: str
) -> list[str]:
    """Replicate the server.py timeout-application logic for testing."""
    resolved_args = list(resolution["arguments"])
    if timeout and lib == "Browser":
        timeout_arg = f"timeout={timeout}"
        if timeout_arg not in resolved_args:
            resolved_args.append(timeout_arg)
    return resolved_args


def test_wait_for_browser_timeout_from_options():
    """wait_for transformer reads timeout from options["timeout"]."""
    adapter = _adapter("Browser")
    result = adapter.resolve_intent(
        intent="wait_for",
        target="id=element",
        options={"timeout": "5s"},
    )
    # The wait_for transformer should embed timeout in args
    assert any("5s" in arg for arg in result["arguments"])


def test_timeout_appended_to_browser_click():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=submit")
    args = _server_apply_timeout(resolution, timeout="10s", lib="Browser")
    assert "timeout=10s" in args


def test_timeout_appended_to_browser_fill():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(
        intent="fill", target="id=name", value="Alice"
    )
    args = _server_apply_timeout(resolution, timeout="10000ms", lib="Browser")
    assert "timeout=10000ms" in args


def test_timeout_not_duplicated():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=submit")
    resolution = dict(resolution)
    resolution["arguments"] = list(resolution["arguments"]) + ["timeout=5s"]
    args = _server_apply_timeout(resolution, timeout="5s", lib="Browser")
    assert args.count("timeout=5s") == 1


def test_timeout_none_no_timeout_arg():
    adapter = _adapter("Browser")
    resolution = adapter.resolve_intent(intent="click", target="id=submit")
    args = _server_apply_timeout(resolution, timeout=None, lib="Browser")
    assert not any("timeout=" in a for a in args)


def test_timeout_selenium_not_appended_as_named_arg():
    """SeleniumLibrary timeout is handled differently; not appended by server."""
    adapter = _adapter("SeleniumLibrary")
    resolution = adapter.resolve_intent(intent="click", target="id=submit")
    args = _server_apply_timeout(resolution, timeout="10s", lib="SeleniumLibrary")
    assert "timeout=10s" not in args
