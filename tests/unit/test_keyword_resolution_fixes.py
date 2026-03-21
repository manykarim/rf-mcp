"""Tests for critical keyword resolution fixes (C1, C3).

Validates that:
- C1: namespace.get_runner() is used instead of the non-existent get_keyword()
- C3: Input Password special case uses unwrapped instances (removed dead path)

Also covers:
- _final_fallback_execution BuiltIn resolution
- Full resolution chain ordering (2-tier: get_runner -> BuiltIn fallback)

Note: C2 (TestLibrary unwrapping) and _try_execute_from_library tests were removed
after ADR-020 F3 eliminated the direct library method lookup path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.components.execution.rf_native_context_manager import (
    RobotFrameworkNativeContextManager,
)

# Correct patch targets:
# - EXECUTION_CONTEXTS is imported at module level in rf_native_context_manager
# - BuiltIn is imported locally inside _final_fallback_execution from robot.libraries.BuiltIn
_EC_PATCH = "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
_BUILTIN_PATCH = "robot.libraries.BuiltIn.BuiltIn"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> RobotFrameworkNativeContextManager:
    return RobotFrameworkNativeContextManager()


def _make_ctx_with_libraries(test_libs):
    """Build a mock EXECUTION_CONTEXTS.current with given libraries."""
    mock_ns = MagicMock()
    mock_ns.libraries = test_libs
    mock_ctx = MagicMock()
    mock_ctx.namespace = mock_ns
    return mock_ctx


class FakeRunner:
    """Mimics the object returned by namespace.get_runner()."""
    def __init__(self, return_value="runner_result"):
        self.return_value = return_value
        self.called = False
        self.call_args = None

    def run(self, data_kw, res_kw, ctx):
        self.called = True
        self.call_args = (data_kw, res_kw, ctx)
        return self.return_value


class FakeNamespace:
    """Fake RF Namespace that supports get_runner()."""
    def __init__(self, runner=None, raise_on_get_runner=False, exc=None):
        self._runner = runner
        self._raise = raise_on_get_runner
        self._exc = exc or RuntimeError("not found")

    def get_runner(self, name, recommend_on_failure=True):
        if self._raise:
            raise self._exc
        return self._runner


# ===========================================================================
# C1: namespace.get_runner() replaces get_keyword()
# ===========================================================================


class TestC1GetRunnerResolution:
    """Verify get_runner() is the primary resolution path."""

    def test_get_runner_is_called_first(self):
        """C1: The first resolution attempt must use namespace.get_runner()."""
        runner = FakeRunner("hello")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "hello"
        assert runner.called

    def test_get_runner_passes_correct_keyword_name(self):
        """C1: The keyword name passed to get_runner() is unmodified."""
        runner = FakeRunner()
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            mgr._execute_any_keyword_generic("New Browser", ["chromium"], ns)

        assert runner.called
        data_kw = runner.call_args[0]
        assert data_kw.name == "New Browser"
        assert tuple(data_kw.args) == ("chromium",)

    def test_get_runner_result_returned_directly(self):
        """C1: Runner result is returned without wrapping."""
        runner = FakeRunner(return_value=42)
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("Evaluate", ["21*2"], ns)

        assert result == 42

    def test_get_runner_failure_falls_through_to_builtin(self):
        """C1: When get_runner() fails, resolution falls through to BuiltIn fallback."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MockBI.return_value
                instance.run_keyword.return_value = "builtin_result"
                result = mgr._execute_any_keyword_generic("New Browser", [], ns)

        assert result == "builtin_result"
        instance.run_keyword.assert_called_once()

    def test_namespace_without_get_runner_skips_to_builtin(self):
        """C1: If namespace has no get_runner() at all, skip to BuiltIn fallback."""
        ns = SimpleNamespace()  # no get_runner attribute
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MockBI.return_value
                instance.run_keyword.return_value = "builtin_result"
                result = mgr._execute_any_keyword_generic("Click", [], ns)

        assert result == "builtin_result"

    def test_get_keyword_is_not_called(self):
        """C1: Verify the removed get_keyword() is never invoked."""
        ns = MagicMock(spec=["get_runner"])
        ns.get_runner.side_effect = RuntimeError("not found")
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = None
            with pytest.raises(RuntimeError):
                mgr._execute_any_keyword_generic("Foo", [], ns)

        # get_keyword must never be called
        assert not hasattr(ns, "get_keyword") or not ns.get_keyword.called


# ===========================================================================
# C3: Input Password special case removed (no longer bypasses unwrapping)
# ===========================================================================


class TestC3InputPasswordSpecialCaseRemoved:
    """Verify the hardcoded Input Password path was removed."""

    def test_input_password_uses_standard_resolution(self):
        """C3: 'Input Password' is resolved via get_runner like any other keyword."""
        runner = FakeRunner("password_set")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic(
                "Input Password", ["locator", "secret"], ns
            )

        assert result == "password_set"
        assert runner.called
        data_kw = runner.call_args[0]
        assert data_kw.name == "Input Password"

    def test_input_password_fallback_uses_builtin(self):
        """C3: When get_runner fails, Input Password falls through to BuiltIn fallback."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MockBI.return_value
                instance.run_keyword.return_value = "done"
                result = mgr._execute_any_keyword_generic(
                    "Input Password", ["loc", "pw"], ns
                )

        assert result == "done"
        instance.run_keyword.assert_called_once()


# ===========================================================================
# _final_fallback_execution
# ===========================================================================


class TestFinalFallbackExecution:
    """Test BuiltIn.run_keyword fallback."""

    def test_builtin_run_keyword_called(self):
        mgr = _make_manager()
        with patch(_BUILTIN_PATCH) as MockBI:
            instance = MockBI.return_value
            instance.run_keyword.return_value = "logged"
            result = mgr._final_fallback_execution("Log", ["message"])
        assert result == "logged"
        instance.run_keyword.assert_called_once_with("Log", "message")

    def test_builtin_direct_method_used_when_run_keyword_fails(self):
        mgr = _make_manager()
        with patch(_BUILTIN_PATCH) as MockBI:
            instance = MockBI.return_value
            instance.run_keyword.side_effect = RuntimeError("fail")
            instance.log = MagicMock(return_value="direct_logged")
            result = mgr._final_fallback_execution("Log", ["message"])
        assert result == "direct_logged"
        instance.log.assert_called_once_with("message")

    def test_raises_when_all_fallbacks_fail(self):
        mgr = _make_manager()
        with patch(_BUILTIN_PATCH) as MockBI:
            instance = MagicMock(spec=[])  # No attributes at all
            instance.run_keyword = MagicMock(side_effect=RuntimeError("nope"))
            MockBI.return_value = instance
            with pytest.raises(RuntimeError, match="could not be resolved"):
                mgr._final_fallback_execution("Nonexistent Keyword", [])


# ===========================================================================
# Full resolution chain ordering
# ===========================================================================


class TestResolutionChainOrder:
    """Verify the 2-tier resolution order: get_runner -> BuiltIn fallback.

    ADR-020 F3 removed the direct library method lookup (path 2), so
    resolution is now: get_runner() -> BuiltIn.run_keyword().
    """

    def test_get_runner_takes_priority_over_builtin_fallback(self):
        """If get_runner succeeds, BuiltIn fallback is never attempted."""
        runner = FakeRunner("runner_wins")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            with patch(_BUILTIN_PATCH) as MockBI:
                result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "runner_wins"
        # BuiltIn should not have been instantiated for resolution
        MockBI.assert_not_called()

    def test_empty_namespace_falls_through_to_builtin(self):
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([])  # empty
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MockBI.return_value
                instance.run_keyword.return_value = "builtin_result"
                result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "builtin_result"

    def test_no_context_falls_through_to_builtin(self):
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = None
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MockBI.return_value
                instance.run_keyword.return_value = "fallback"
                result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "fallback"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and regression guards."""

    def test_empty_arguments(self):
        runner = FakeRunner("no_args_ok")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("Close Browser", [], ns)

        assert result == "no_args_ok"
        data_kw = runner.call_args[0]
        assert tuple(data_kw.args) == ()

    def test_keyword_with_dots_in_name(self):
        runner = FakeRunner("qualified_ok")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("Browser.Click", ["loc"], ns)

        assert result == "qualified_ok"

    def test_runner_returning_none_is_valid(self):
        runner = FakeRunner(return_value=None)
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result is None
        assert runner.called

    def test_get_runner_failure_falls_to_builtin_then_raises(self):
        """When get_runner fails and BuiltIn fallback also fails, raises RuntimeError."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MagicMock(spec=[])  # No attributes -- no direct method fallback
                instance.run_keyword = MagicMock(side_effect=RuntimeError("also fails"))
                MockBI.return_value = instance
                with pytest.raises(RuntimeError, match="could not be resolved"):
                    mgr._execute_any_keyword_generic("Click", ["loc"], ns)

    def test_many_arguments(self):
        runner = FakeRunner("many_args_ok")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        args = [f"arg{i}" for i in range(20)]

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("Create Dictionary", args, ns)

        assert result == "many_args_ok"
        data_kw = runner.call_args[0]
        assert len(data_kw.args) == 20

    def test_single_char_keyword(self):
        """Single-character keyword names work."""
        runner = FakeRunner("x_result")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            result = mgr._execute_any_keyword_generic("X", [], ns)

        assert result == "x_result"
