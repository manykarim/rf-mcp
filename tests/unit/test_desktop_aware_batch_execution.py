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


# ── §4/§5: retry-safety gate + descriptor-timeout cap ──────────────────────
class _FakeSession:
    def __init__(self, desktop):
        self._d = desktop
    def is_desktop_session(self):
        return self._d


class _FakeSessionManager:
    def __init__(self, desktop):
        self._s = _FakeSession(desktop)
    def get_session(self, sid):
        return self._s


def _adapter(desktop):
    from robotmcp.adapters.recovery_adapter import RecoveryServiceAdapter
    from robotmcp.domains.recovery.aggregates import RecoveryEngine

    class _KR:
        async def run_keyword(self, *a, **k):
            return None
    return RecoveryServiceAdapter(
        engine=RecoveryEngine.with_defaults(),
        keyword_runner=_KR(),
        session_manager=_FakeSessionManager(desktop),
    )


class TestDesktopRetryGate:
    def test_desktop_non_element_not_found_is_blocked(self):
        a = _adapter(desktop=True)
        # a post-action / timeout-ish error is NOT ELEMENT_NOT_FOUND -> blocked
        assert a.desktop_retry_blocked("s", "Timeout exceeded waiting") is True

    def test_desktop_element_not_found_allowed(self):
        a = _adapter(desktop=True)
        assert a.desktop_retry_blocked(
            "s", "ElementNotFoundError: No UiNode found for descriptor after timeout of 30 seconds"
        ) is False

    def test_web_never_blocked(self):
        a = _adapter(desktop=False)
        assert a.desktop_retry_blocked("s", "Timeout exceeded") is False


class TestRetryTimeoutCap:
    def test_cap_seconds_env(self, monkeypatch):
        from robotmcp.adapters.recovery_adapter import _batch_retry_timeout_cap_seconds
        monkeypatch.delenv("ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT", raising=False)
        assert _batch_retry_timeout_cap_seconds() == 5.0
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT", "9")
        assert _batch_retry_timeout_cap_seconds() == 9.0
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT", "bad")
        assert _batch_retry_timeout_cap_seconds() == 5.0

    def test_cap_applied_and_restored_on_desktop(self, monkeypatch):
        a = _adapter(desktop=True)

        class _QS:
            timeout = 30.0
        fake_lib = type("L", (), {"query_settings": _QS()})()
        monkeypatch.setattr(a, "_resolve_baremetal", lambda sid: fake_lib)
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT", "5")
        with a.retry_timeout_cap("s"):
            assert fake_lib.query_settings.timeout == 5.0
        assert fake_lib.query_settings.timeout == 30.0  # restored

    def test_cap_noop_when_no_library(self, monkeypatch):
        a = _adapter(desktop=True)
        monkeypatch.setattr(a, "_resolve_baremetal", lambda sid: None)
        with a.retry_timeout_cap("s"):
            pass  # must not raise

    def test_cap_noop_on_web(self, monkeypatch):
        a = _adapter(desktop=False)
        called = {"n": 0}
        def _rb(sid):
            called["n"] += 1
            return None
        monkeypatch.setattr(a, "_resolve_baremetal", _rb)
        with a.retry_timeout_cap("s"):
            pass
        assert called["n"] == 0  # web never resolves the library


class TestBatchRunnerDesktopGate:
    def test_blocked_desktop_failure_does_not_retry(self):
        import asyncio
        from robotmcp.domains.batch_execution.services import BatchRunner
        from robotmcp.domains.batch_execution.aggregates import BatchExecution

        calls = {"n": 0}

        class _KWExec:
            async def execute_keyword(self, sid, kw, args, timeout=None, assign_to=None):
                calls["n"] += 1
                return {"success": False, "error": "Timeout exceeded"}

        class _Recovery:
            supports_desktop_batch_hooks = True
            def desktop_retry_blocked(self, sid, err):
                return True  # simulate desktop, non-ELEMENT_NOT_FOUND
            async def attempt_recovery(self, *a, **k):
                return None

        batch = BatchExecution.create(
            "s", [{"keyword": "Pointer Click", "arguments": ["/app:*//x"]}],
            on_failure="recover", max_recovery_attempts=2,
        )
        runner = BatchRunner(keyword_executor=_KWExec(), recovery_service=_Recovery())
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(runner.execute(batch))
        finally:
            loop.close()
        # Only the initial attempt ran; the gate blocked all retries.
        assert calls["n"] == 1

    def test_unblocked_failure_retries(self):
        import asyncio
        from robotmcp.domains.batch_execution.services import BatchRunner
        from robotmcp.domains.batch_execution.aggregates import BatchExecution

        calls = {"n": 0}

        class _KWExec:
            async def execute_keyword(self, sid, kw, args, timeout=None, assign_to=None):
                calls["n"] += 1
                return {"success": False, "error": "element not found"}

        class _Recovery:
            supports_desktop_batch_hooks = True
            def desktop_retry_blocked(self, sid, err):
                return False  # allowed (web, or ELEMENT_NOT_FOUND)
            async def attempt_recovery(self, *a, **k):
                return None

        batch = BatchExecution.create(
            "s", [{"keyword": "Click", "arguments": ["x"]}],
            on_failure="recover", max_recovery_attempts=2,
        )
        runner = BatchRunner(keyword_executor=_KWExec(), recovery_service=_Recovery())
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(runner.execute(batch))
        finally:
            loop.close()
        # initial + 2 retries
        assert calls["n"] == 3
