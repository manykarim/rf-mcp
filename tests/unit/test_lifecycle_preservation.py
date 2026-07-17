"""Active-test state survives RF context recreation
(change: desktop-evidence-and-display-scoping, D6).

The 2026-06-11 rerun hit "No active test to end" after start_test: the
build_test_suite dry-run path re-runs create_context_for_session, whose
None-initialized _initial_*_test values clobbered the stored
current_run_test / current_res_test. The fix seeds them from the existing
session-context entry.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)


@pytest.fixture()
def mgr():
    return get_rf_native_context_manager()


class TestContextRecreationPreservesActiveTest:
    def test_recreate_preserves_current_test_entries(self, mgr):
        sid = "lifecycle-1"
        res = mgr.create_context_for_session(sid, ["BuiltIn"])
        assert res.get("success") is True
        started = mgr.start_test_in_context(sid, "My Active Test")
        assert started.get("success") is True
        entry = mgr._session_contexts[sid]
        assert entry["current_res_test"] is not None

        # Simulate the build_test_suite dry-run refresh: same session,
        # context already exists.
        res2 = mgr.create_context_for_session(sid, ["BuiltIn"])
        assert res2.get("success") is True
        entry2 = mgr._session_contexts[sid]
        assert entry2["current_res_test"] is not None, (
            "context refresh must not clobber the active test"
        )

        ended = mgr.end_test_in_context(sid)
        assert ended.get("success") is True, ended

    def test_start_build_end_bracket(self, mgr):
        # The full Run-3 failure shape: start_test → context refresh →
        # end_test must succeed (no "No active test to end").
        sid = "lifecycle-2"
        mgr.create_context_for_session(sid, ["BuiltIn"])
        mgr.start_test_in_context(sid, "Writer Validation")
        mgr.create_context_for_session(sid, ["BuiltIn"])  # dry-run refresh
        ended = mgr.end_test_in_context(sid)
        assert ended.get("success") is True
        assert "No active test" not in str(ended)

    def test_fresh_session_still_starts_clean(self, mgr):
        # No prior entry: initial test references stay None (unchanged
        # behavior for first-time context creation).
        sid = "lifecycle-3"
        mgr._session_contexts.pop(sid, None)
        mgr.create_context_for_session(sid, ["BuiltIn"])
        entry = mgr._session_contexts[sid]
        # Either None (minimal context) or a synthetic initial test —
        # but end_test without start_test must fail cleanly.
        if entry["current_res_test"] is None:
            ended = mgr.end_test_in_context(sid)
            assert ended.get("success") is False
