"""Desktop-aware batch execution (change: desktop-aware-batch-execution).

execute_batch is now in the desktop_exec profile, PlatynUI element-resolution
errors classify as ELEMENT_NOT_FOUND (not TIMEOUT), and batch recovery is
platform-filtered so a desktop session never gets browser recovery actions.
"""

from __future__ import annotations

import pytest

from robotmcp.domains.recovery.aggregates import RecoveryEngine
from robotmcp.domains.recovery.value_objects import (
    ErrorClassification,
    RecoveryStrategy,
    RecoveryTier,
)

_BROWSER_ACTIONS = {"Execute Javascript", "Reload Page", "Go Back", "Handle Alert"}


class TestDesktopProfileHasBatch:
    def test_desktop_exec_includes_execute_batch(self):
        from robotmcp.domains.tool_profile.aggregates import ProfilePresets

        profile = ProfilePresets.desktop_exec()
        assert "execute_batch" in profile.tool_names


class TestPlatynUIClassification:
    def test_platynui_error_classifies_element_not_found_not_timeout(self):
        e = RecoveryEngine.with_defaults()
        # the real PlatynUI message embeds "timeout of 30 seconds"
        c = e.classify(
            "ElementNotFoundError: No UiNode found for descriptor after timeout of 30 seconds"
        )
        assert c is ErrorClassification.ELEMENT_NOT_FOUND

    def test_browser_error_corpus_unchanged(self):
        e = RecoveryEngine.with_defaults()
        assert e.classify("element not found") is ErrorClassification.ELEMENT_NOT_FOUND
        assert e.classify("Timeout exceeded") is ErrorClassification.TIMEOUT_EXCEPTION
        assert e.classify("click intercepted") is ErrorClassification.ELEMENT_CLICK_INTERCEPTED


class TestPlatformAwareStrategies:
    def test_strategy_defaults_to_web(self):
        s = RecoveryStrategy(name="x", tier=RecoveryTier.TIER_1)
        assert s.applies_to_platform("web") and not s.applies_to_platform("desktop")

    def test_desktop_selection_never_returns_browser_actions(self):
        e = RecoveryEngine.with_defaults()
        acts = set()
        for n in (1, 2, 3):
            s = e.select_strategy(ErrorClassification.ELEMENT_NOT_FOUND, n, platform="desktop")
            if s:
                acts.update(a.keyword for a in s.actions)
        assert acts and not (acts & _BROWSER_ACTIONS)

    def test_desktop_tiers(self):
        e = RecoveryEngine.with_defaults()
        assert e.select_strategy(ErrorClassification.ELEMENT_NOT_FOUND, 1, platform="desktop").name == "desktop_wait_and_retry"
        assert e.select_strategy(ErrorClassification.ELEMENT_NOT_FOUND, 2, platform="desktop").name == "desktop_activate_window"

    def test_web_selection_unchanged(self):
        e = RecoveryEngine.with_defaults()
        # default platform is web; existing behavior preserved
        assert e.select_strategy(ErrorClassification.ELEMENT_NOT_FOUND, 1).name == "wait_and_retry"
        assert e.select_strategy(ErrorClassification.ELEMENT_NOT_FOUND, 1, platform="web").name == "wait_and_retry"


class TestBatchSteer:
    def test_desktop_guidance_steers_to_batch(self):
        from robotmcp.components.execution.desktop_guidance import get_desktop_guidance

        b = get_desktop_guidance()
        assert "execute_batch" in b.get("batching", "")
