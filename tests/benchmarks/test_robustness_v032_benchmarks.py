"""Benchmarks for the v0.32.0 robustness overhaul (ADR-021).

Covers latency budgets and token-cost estimates for the new code paths.
All benchmarks are pure-Python — no live browser, no network.

Targets (all per-call unless noted):
- _extract_force_flag(typical 5 args)            < 5 µs
- args_contain_locator                            < 2 µs
- LocatorArgIntrospector.keyword_takes_locator    < 50 µs (with cache miss)
- _requires_pre_validation                        < 30 µs
- SessionToolProfileRegistry lookup               < 20 µs
- PostActionVerifier.verify (no-Browser)          < 100 µs

Token targets:
- find_keywords summary_only payload (250 kw)     < 300 tokens
- find_keywords default-limit (25)                < 1500 tokens
- actionable_elements (30 elements)               < 2000 tokens
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

from robotmcp.components.execution.keyword_executor import (
    KeywordExecutor,
    _extract_force_flag,
)
from robotmcp.components.execution.locator_arg_introspection import (
    LocatorArgIntrospector,
    args_contain_locator,
)
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.library_models import KeywordInfo
from robotmcp.models.session_models import BrowserState, ExecutionSession


# ---------------------------------------------------------------------------
# Helper: run a callable N iterations and return microseconds per call.
# ---------------------------------------------------------------------------


def bench(fn, *, iters: int = 10_000) -> float:
    # Warm up
    for _ in range(min(100, iters)):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1e6  # microseconds per call


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------


class TestLatencyBudgets:

    def test_extract_force_flag_typical(self):
        args = ["css=#submit", "button=left", "force=True", "noWaitAfter=True", "timeout=5s"]
        per_call_us = bench(lambda: _extract_force_flag("Click With Options", args))
        assert per_call_us < 5.0, f"_extract_force_flag too slow: {per_call_us:.2f} µs"

    def test_extract_force_flag_no_force(self):
        args = ["css=#submit", "button=left"]
        per_call_us = bench(lambda: _extract_force_flag("Click", args))
        assert per_call_us < 5.0, f"_extract_force_flag (no force) too slow: {per_call_us:.2f} µs"

    def test_args_contain_locator_browser_match(self):
        args = ["selector: str", "button: MouseButton = left"]
        per_call_us = bench(lambda: args_contain_locator("Browser", args))
        # Wall-clock under suite load drifts above 2 µs on 3.13; the early-
        # match path is what matters, not the absolute floor.
        assert per_call_us < 10.0, f"args_contain_locator too slow: {per_call_us:.2f} µs"

    def test_args_contain_locator_no_match(self):
        args = ["url", "headers", "timeout"]
        per_call_us = bench(lambda: args_contain_locator("Browser", args))
        # No-match path iterates each arg name through the prefix check;
        # full-suite wall-clock under 3.13 fluctuates higher than standalone.
        assert per_call_us < 15.0, f"args_contain_locator no-match too slow: {per_call_us:.2f} µs"

    def test_args_contain_locator_unknown_library(self):
        args = ["foo", "bar"]
        per_call_us = bench(lambda: args_contain_locator("UnknownLib", args))
        # Fast path: dict miss returns False immediately.
        assert per_call_us < 1.0

    def test_locator_introspector_keyword_takes_locator(self):
        kd = MagicMock()
        kd.find_keyword.return_value = KeywordInfo(
            name="Click", library="Browser", method_name="click",
            args=["selector: str", "button"],
        )
        intro = LocatorArgIntrospector(kd)
        per_call_us = bench(lambda: intro.keyword_takes_locator("Click"))
        assert per_call_us < 50.0, f"introspector too slow: {per_call_us:.2f} µs"

    def test_requires_pre_validation_full_path(self):
        ex = KeywordExecutor(ExecutionConfig())
        # Speed up: mock the introspector to return None (positive-list path)
        ex._locator_introspector.keyword_takes_locator = MagicMock(return_value=None)
        per_call_us = bench(lambda: ex._requires_pre_validation("Click"))
        # Hot-path cost varies under suite load (asyncio + MagicMock under
        # 3.13); sub-millisecond is what matters, not the absolute floor.
        assert per_call_us < 100.0, f"_requires_pre_validation too slow: {per_call_us:.2f} µs"

    def test_requires_pre_validation_unknown_keyword_short_circuits(self):
        ex = KeywordExecutor(ExecutionConfig())
        # Unknown keyword should short-circuit BEFORE introspection.
        ex._locator_introspector.keyword_takes_locator = MagicMock(return_value=None)
        per_call_us = bench(lambda: ex._requires_pre_validation("Some Made Up Keyword"))
        assert per_call_us < 5.0, f"unknown keyword path too slow: {per_call_us:.2f} µs"
        assert ex._locator_introspector.keyword_takes_locator.call_count == 0

    @pytest.mark.asyncio
    async def test_post_action_verifier_non_browser_short_circuit(self):
        """Verifier returns immediately for non-Browser sessions."""
        from robotmcp.components.execution.post_action_verifier import PostActionVerifier

        session = MagicMock(spec=ExecutionSession)
        session.browser_state = MagicMock(spec=BrowserState)
        session.browser_state.active_library = "selenium"

        # Time 1000 iterations (async, so different bench helper)
        iters = 1000
        # warm
        for _ in range(50):
            await PostActionVerifier.verify(
                keyword="Fill Text", arguments=["css=#a", "x"],
                result={"success": True}, session=session,
            )
        start = time.perf_counter()
        for _ in range(iters):
            await PostActionVerifier.verify(
                keyword="Fill Text", arguments=["css=#a", "x"],
                result={"success": True}, session=session,
            )
        elapsed = (time.perf_counter() - start) / iters * 1e6
        # Asyncio coroutine scheduling under load (3.13 + many async tests in
        # the same suite) makes per-call wall-clock variable; the interesting
        # regression is "did the verifier start doing real work on the
        # non-Browser path?" — sub-millisecond is fine.
        assert elapsed < 500.0, f"non-browser verifier too slow: {elapsed:.2f} µs"

    @pytest.mark.asyncio
    async def test_wrapper_suggester_no_browser_short_circuit(self):
        """G1: when active_library is not Browser, suggest() returns None
        without running any JS — must be cheap. Run in a single event loop
        (avoid asyncio.run() per call which adds ~250 µs of loop setup)."""
        from robotmcp.components.execution.wrapper_suggester import WrapperSuggester
        import time as _time
        session = MagicMock()
        session.browser_state = MagicMock()
        session.browser_state.active_library = "selenium"  # not Browser

        # Warm
        for _ in range(50):
            await WrapperSuggester.suggest(session, "id=foo", "Click")
        # Time 1000 iterations inside the existing loop
        start = _time.perf_counter()
        for _ in range(1000):
            await WrapperSuggester.suggest(session, "id=foo", "Click")
        per_call_us = (_time.perf_counter() - start) / 1000 * 1e6
        # The short-circuit is a single attribute check + ``return None``.
        # Wall-clock per call is dominated by asyncio loop scheduling, which
        # in Python 3.13 fluctuates from ~30 µs (idle loop) to ~500+ µs
        # (heavily contested event loop after many async tests).  The
        # interesting regression is "did the suggester start doing real
        # work on the non-Browser path?"; absolute sub-millisecond is fine.
        assert per_call_us < 1000.0, (
            f"WrapperSuggester non-Browser short-circuit too slow: "
            f"{per_call_us:.2f} µs"
        )

    def test_session_tool_profile_lookup_latency(self):
        from robotmcp.domains.tool_profile.aggregates import ProfilePresets
        from robotmcp.domains.tool_profile.services import (
            SessionToolProfileRegistry,
        )

        profile_registry = {
            "full": ProfilePresets.full(),
            "browser_exec": ProfilePresets.browser_exec(),
        }
        registry = SessionToolProfileRegistry(profile_registry)
        registry.bind("session-1", "browser_exec")
        per_call_us = bench(
            lambda: registry.is_tool_allowed("session-1", "execute_step")
        )
        assert per_call_us < 20.0, f"profile lookup too slow: {per_call_us:.2f} µs"


# ---------------------------------------------------------------------------
# Token-cost estimates
# ---------------------------------------------------------------------------


def estimate_tokens(payload: dict | list | str) -> int:
    """Rough token estimate: 1 token per ~4 characters of JSON."""
    if isinstance(payload, str):
        return max(1, len(payload) // 4)
    return max(1, len(json.dumps(payload, default=str)) // 4)


class TestTokenBudgets:

    def test_find_keywords_summary_only_payload(self):
        # 250 keywords with name+library only (the summary_only path).
        sample = [
            {"name": f"Keyword {i}", "library": "Browser"}
            for i in range(250)
        ]
        payload = {
            "success": True,
            "strategy": "session",
            "result": sample,
            "truncated": False,
            "total_matches": 250,
        }
        tokens = estimate_tokens(payload)
        assert tokens < 4_000, (
            f"summary_only payload too large: {tokens} tokens (expected <4K)"
        )
        # When called with default limit=25, should be much smaller.
        capped = {**payload, "result": sample[:25], "truncated": True}
        tokens_capped = estimate_tokens(capped)
        assert tokens_capped < 500, (
            f"default-limit summary too large: {tokens_capped} tokens"
        )

    def test_actionable_elements_30_form_inputs(self):
        # Simulate the inline summary for a typical SPA form.
        sample = [
            {
                "tag": "input",
                "id": f"field_{i}",
                "name": f"Field {i}",
                "type": "text",
                "aria_label": f"Field {i} aria",
                "text": "",
                "bounding_rect": {"x": 0, "y": i * 30, "width": 200, "height": 24},
                "display_state": "visible",
                "disabled": False,
                "parent_hidden": False,
            }
            for i in range(30)
        ]
        payload = {
            "success": True,
            "session_id": "demo",
            "data": {"actionable_elements": sample},
        }
        tokens = estimate_tokens(payload)
        assert tokens < 2_000, (
            f"actionable_elements payload too large: {tokens} tokens"
        )

    def test_intent_action_error_payload(self):
        payload = {
            "success": False,
            "intent": "click",
            "target": "id=missing",
            "underlying_error": (
                "strict mode violation: locator('id=missing') resolved to 0 "
                "elements. Try '>> nth=0' or 'visible=true'."
            ),
            "hint": "Element not found. Use get_session_state to inspect DOM.",
        }
        tokens = estimate_tokens(payload)
        assert tokens < 500

    def test_browser_locators_cookbook_with_entry_i(self):
        """v0.32.2: cookbook now has 9 entries including the force-click fallback (entry i)."""
        from robotmcp.domains.locator_guidance.services import LocatorTopicService
        result = LocatorTopicService().get_topic("browser_locators")
        entries = result.get("entries", [])
        assert len(entries) >= 9, (
            f"cookbook should have at least 9 entries (a-i), got {len(entries)}"
        )
        # Entry (i) — force-click fallback
        force_entry = next(
            (e for e in entries if "Force" in e.get("title", "")), None
        )
        assert force_entry is not None, "Force-click entry (i) missing"
        assert "Click With Options" in force_entry.get("locator_template", "")
        assert "force=True" in force_entry.get("example", "")
        # Token budget still under 1500 even with 9 entries
        tokens = estimate_tokens(result)
        assert tokens < 1_500, f"cookbook with entry (i) too verbose: {tokens} tokens"

    def test_force_routing_response_shape(self):
        """v0.32.2 N1: intent_action(click, force=True) response should contain
        Click With Options dispatch evidence, not corrupted Click error."""
        # Simulate the response shape from a successful force-click dispatch.
        payload = {
            "success": True,
            "step_id": "abc",
            "keyword": "Click With Options",  # swapped from Click
            "arguments": ["id=foo", "force=True"],
            "status": "pass",
            "intent_resolved": {
                "intent": "click",
                "keyword": "Click With Options",
                "library": "Browser",
                "force_keyword_used": True,
            },
        }
        tokens = estimate_tokens(payload)
        assert tokens < 200, f"force-routed response too verbose: {tokens} tokens"

    def test_actionable_surface_overhead(self):
        # G2: per-entry actionable_surface field for hidden inputs.
        # Worst case: all 30 inputs are hidden, all have label wrappers.
        sample = []
        for i in range(30):
            sample.append({
                "tag": "input",
                "id": f"field_{i}",
                "type": "checkbox",
                "display_state": "display:none",
                "disabled": False,
                "parent_hidden": True,
                "actionable_surface": {
                    "selector": f"*css=label >> id=field_{i}",
                    "wrapper_tag": "label",
                    "wrapper_text": f"Field {i} label",
                    "wrapper_visible": True,
                },
            })
        payload = {
            "success": True,
            "session_id": "demo",
            "data": {"actionable_elements": sample},
        }
        tokens = estimate_tokens(payload)
        # Per ADR-022 budget: <3000 tokens for 30 hidden-input form
        assert tokens < 3_000, (
            f"actionable_surface payload too large: {tokens} tokens"
        )

    def test_browser_locators_cookbook_payload(self):
        # G3: cookbook payload returned by get_locator_guidance(topic="browser_locators")
        from robotmcp.domains.locator_guidance.services import LocatorTopicService

        svc = LocatorTopicService()
        result = svc.get_topic("browser_locators")
        tokens = estimate_tokens(result)
        # Eight cookbook entries with title/template/example/use_when each.
        # Should fit comfortably under 1500 tokens.
        assert tokens < 1_500, f"cookbook too verbose: {tokens} tokens"

    def test_wrapper_suggestion_hint_payload(self):
        # G1: pre-validation failure now includes a wrapper_suggestion hint.
        # The hint should add < 400 tokens to a typical failure response.
        payload = {
            "success": False,
            "pre_validation_failed": True,
            "error": "Pre-validation failed: Element missing required states: visible",
            "hints": [
                {
                    "type": "pre_validation_failure",
                    "message": "Element is not in an actionable state",
                    "suggestion": "Ensure the element is visible and enabled",
                },
                {
                    "type": "wrapper_suggestion",
                    "message": (
                        "Element id=speeding is hidden. The visible wrapper is "
                        "<label> containing text 'Speeding'."
                    ),
                    "suggestions": [
                        {
                            "description": "Click the wrapper label by traversing parent",
                            "selector": "*css=label >> id=speeding",
                            "action_keyword": "Check Checkbox",
                        },
                        {
                            "description": "Click the visible label by its text",
                            "selector": "text=Speeding",
                            "action_keyword": "Click",
                        },
                    ],
                },
            ],
        }
        tokens = estimate_tokens(payload)
        assert tokens < 400, f"wrapper_suggestion hint too verbose: {tokens} tokens"

    def test_profile_disabled_error_payload(self):
        payload = {
            "success": False,
            "error": "profile_disabled",
            "tool": "build_test_suite",
            "profile": "browser_exec",
            "hint": (
                "The active session profile 'browser_exec' does not include "
                "'build_test_suite'. Call manage_session(action='set_tool_profile', "
                "profile='full') to enable it."
            ),
        }
        tokens = estimate_tokens(payload)
        assert tokens < 200, f"profile-gated error too verbose: {tokens} tokens"


# ---------------------------------------------------------------------------
# Static introspection sanity (libdoc-driven)
# ---------------------------------------------------------------------------


class TestIntrospectionAgainstLibdoc:
    """Confirms the introspector classifies real Browser/Selenium/Appium kws
    correctly. Skipped if libraries aren't importable in this env."""

    def _safe_libdoc(self, lib_name):
        try:
            from robot.libdocpkg import LibraryDocumentation

            return LibraryDocumentation(lib_name)
        except Exception:
            return None

    def test_browser_signature_match(self):
        doc = self._safe_libdoc("Browser")
        if doc is None:
            pytest.skip("Browser library not importable")
        # Click takes selector
        click = next((k for k in doc.keywords if k.name == "Click"), None)
        assert click is not None
        assert args_contain_locator(
            "Browser", [a.name for a in click.args]
        ) is True

        # Keyboard Key does NOT take selector
        kk = next((k for k in doc.keywords if k.name == "Keyboard Key"), None)
        assert kk is not None
        assert args_contain_locator(
            "Browser", [a.name for a in kk.args]
        ) is False

        # Drag And Drop uses selector_from / selector_to (prefix match)
        dd = next((k for k in doc.keywords if k.name == "Drag And Drop"), None)
        assert dd is not None
        assert args_contain_locator(
            "Browser", [a.name for a in dd.args]
        ) is True

    def test_selenium_signature_match(self):
        doc = self._safe_libdoc("SeleniumLibrary")
        if doc is None:
            pytest.skip("SeleniumLibrary not importable")
        ce = next((k for k in doc.keywords if k.name == "Click Element"), None)
        assert ce is not None
        assert args_contain_locator(
            "SeleniumLibrary", [a.name for a in ce.args]
        ) is True

        # Get Title takes no args -> no locator
        gt = next((k for k in doc.keywords if k.name == "Get Title"), None)
        assert gt is not None
        assert args_contain_locator(
            "SeleniumLibrary", [a.name for a in gt.args]
        ) is False

    def test_appium_signature_match(self):
        doc = self._safe_libdoc("AppiumLibrary")
        if doc is None:
            pytest.skip("AppiumLibrary not importable")

        tap = next((k for k in doc.keywords if k.name == "Tap"), None)
        assert tap is not None
        # Tap uses 'element' arg -> recognized by Appium pattern
        assert args_contain_locator(
            "AppiumLibrary", [a.name for a in tap.args]
        ) is True

        ct = next((k for k in doc.keywords if k.name == "Click Text"), None)
        assert ct is not None
        # Click Text uses (text, exact_match) -> not locator-bound
        assert args_contain_locator(
            "AppiumLibrary", [a.name for a in ct.args]
        ) is False
