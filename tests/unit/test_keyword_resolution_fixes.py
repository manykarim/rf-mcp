"""Tests for critical keyword resolution fixes (C1, C2, C3).

Validates that:
- C1: namespace.get_runner() is used instead of the non-existent get_keyword()
- C2: TestLibrary wrappers are unwrapped to raw instances in library search
- C3: Input Password special case uses unwrapped instances (removed dead path)

Also covers:
- _try_execute_from_library naming conventions
- _final_fallback_execution BuiltIn resolution
- Full resolution chain ordering
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


class FakeTestLibrary:
    """Mimics RF 7 TestLibrary wrapper."""
    def __init__(self, name: str, instance):
        self.name = name
        self._instance = instance

    @property
    def instance(self):
        return self._instance


class FakeLibraryInstance:
    """A raw Python library instance with callable methods."""
    def __init__(self, methods: dict):
        for name, func in methods.items():
            setattr(self, name, func)


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

    def test_get_runner_failure_falls_through_to_library_search(self):
        """C1: When get_runner() fails, resolution continues to library search."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        called_method = MagicMock(return_value="lib_result")
        raw_lib = FakeLibraryInstance({"new_browser": called_method})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            result = mgr._execute_any_keyword_generic("New Browser", [], ns)

        assert result == "lib_result"
        called_method.assert_called_once_with()

    def test_namespace_without_get_runner_skips_to_library_search(self):
        """C1: If namespace has no get_runner() at all, skip gracefully."""
        ns = SimpleNamespace()  # no get_runner attribute
        mgr = _make_manager()

        called = MagicMock(return_value="direct")
        raw_lib = FakeLibraryInstance({"click": called})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            result = mgr._execute_any_keyword_generic("Click", [], ns)

        assert result == "direct"

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
# C2: TestLibrary wrappers unwrapped in library search
# ===========================================================================


class TestC2TestLibraryUnwrapping:
    """Verify TestLibrary.instance is used, not the wrapper itself."""

    def test_library_instance_is_unwrapped(self):
        """C2: _try_execute_from_library receives the raw Python instance."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        clicked = MagicMock(return_value="clicked!")
        raw_lib = FakeLibraryInstance({"click": clicked})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            result = mgr._execute_any_keyword_generic("Click", [], ns)

        assert result == "clicked!"
        clicked.assert_called_once()

    def test_library_name_uses_test_library_name_attribute(self):
        """C2: Library name comes from TestLibrary.name, not __class__.__name__."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        method = MagicMock(return_value="ok")
        raw_lib = FakeLibraryInstance({"new_browser": method})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            with patch("robotmcp.components.execution.rf_native_context_manager.logger") as log:
                mgr._execute_any_keyword_generic("New Browser", [], ns)
                log_calls = " ".join(str(c) for c in log.info.call_args_list)
                assert "Browser" in log_calls

    def test_wrapper_without_instance_attribute_used_directly(self):
        """C2: If object has no .instance, getattr fallback uses the object itself."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        # Object that IS the library (no .instance wrapper) — simulates old-style
        raw_lib = FakeLibraryInstance({"log": MagicMock(return_value="logged")})
        raw_lib.name = "BuiltIn"
        # getattr(raw_lib, 'instance', raw_lib) returns raw_lib since no .instance

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([raw_lib])
            result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "logged"

    def test_multiple_libraries_searched_in_order(self):
        """C2: All libraries are searched; first match wins."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        builtin_log = MagicMock(return_value="builtin_result")
        browser_log = MagicMock(return_value="browser_result")

        builtin = FakeLibraryInstance({"log": builtin_log})
        browser = FakeLibraryInstance({"log": browser_log})

        test_libs = [
            FakeTestLibrary("BuiltIn", builtin),
            FakeTestLibrary("Browser", browser),
        ]

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries(test_libs)
            result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "builtin_result"
        builtin_log.assert_called_once()
        browser_log.assert_not_called()

    def test_library_without_matching_keyword_skipped(self):
        """C2: Libraries that don't have the keyword are skipped."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        click_method = MagicMock(return_value="clicked")
        builtin = FakeLibraryInstance({})  # no "click"
        browser = FakeLibraryInstance({"click": click_method})

        test_libs = [
            FakeTestLibrary("BuiltIn", builtin),
            FakeTestLibrary("Browser", browser),
        ]

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries(test_libs)
            result = mgr._execute_any_keyword_generic("Click", [], ns)

        assert result == "clicked"


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

    def test_input_password_fallback_uses_unwrapped_instance(self):
        """C3: When get_runner fails, Input Password finds the method on the raw instance."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        ip_method = MagicMock(return_value="done")
        raw_selenium = FakeLibraryInstance({"input_password": ip_method})
        test_lib = FakeTestLibrary("SeleniumLibrary", raw_selenium)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            result = mgr._execute_any_keyword_generic(
                "Input Password", ["loc", "pw"], ns
            )

        assert result == "done"
        ip_method.assert_called_once_with("loc", "pw")


# ===========================================================================
# _try_execute_from_library naming conventions
# ===========================================================================


class TestTryExecuteFromLibrary:
    """Test method name normalization in _try_execute_from_library."""

    def test_canonical_snake_case(self):
        """'New Browser' -> 'new_browser'."""
        mgr = _make_manager()
        m = MagicMock(return_value="ok")
        lib = FakeLibraryInstance({"new_browser": m})
        result = mgr._try_execute_from_library("New Browser", ["arg"], "Browser", lib)
        assert result == "ok"
        m.assert_called_once_with("arg")

    def test_preserved_case_underscore(self):
        """'New Browser' -> 'New_Browser' (alternative naming)."""
        mgr = _make_manager()
        m = MagicMock(return_value="ok2")
        lib = FakeLibraryInstance({"New_Browser": m})
        result = mgr._try_execute_from_library("New Browser", [], "Browser", lib)
        assert result == "ok2"

    def test_concatenated_lowercase(self):
        """'New Browser' -> 'newbrowser' (no separator)."""
        mgr = _make_manager()
        m = MagicMock(return_value="ok3")
        lib = FakeLibraryInstance({"newbrowser": m})
        result = mgr._try_execute_from_library("New Browser", [], "Browser", lib)
        assert result == "ok3"

    def test_no_matching_method_returns_none(self):
        mgr = _make_manager()
        lib = FakeLibraryInstance({})
        result = mgr._try_execute_from_library("Click", [], "Browser", lib)
        assert result is None

    def test_non_callable_attribute_skipped(self):
        mgr = _make_manager()
        lib = FakeLibraryInstance({})
        lib.click = "not_callable"
        result = mgr._try_execute_from_library("Click", [], "Browser", lib)
        assert result is None

    def test_arguments_passed_through(self):
        mgr = _make_manager()
        m = MagicMock(return_value="result")
        lib = FakeLibraryInstance({"fill_text": m})
        mgr._try_execute_from_library("Fill Text", ["loc", "value", "force=True"], "Browser", lib)
        m.assert_called_once_with("loc", "value", "force=True")

    def test_canonical_name_preferred_over_alternatives(self):
        """When canonical and alternative both exist, canonical wins."""
        mgr = _make_manager()
        canonical = MagicMock(return_value="canon")
        alternative = MagicMock(return_value="alt")
        lib = FakeLibraryInstance({"new_browser": canonical, "New_Browser": alternative})
        result = mgr._try_execute_from_library("New Browser", [], "Browser", lib)
        assert result == "canon"
        canonical.assert_called_once()
        alternative.assert_not_called()


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
    """Verify the 3-tier resolution order: get_runner -> library search -> BuiltIn."""

    def test_get_runner_takes_priority_over_library_search(self):
        """If get_runner succeeds, library search is never attempted."""
        runner = FakeRunner("runner_wins")
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        click_method = MagicMock(return_value="lib_wins")
        raw_lib = FakeLibraryInstance({"log": click_method})
        test_lib = FakeTestLibrary("BuiltIn", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            result = mgr._execute_any_keyword_generic("Log", ["msg"], ns)

        assert result == "runner_wins"
        click_method.assert_not_called()

    def test_library_search_takes_priority_over_builtin_fallback(self):
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        method = MagicMock(return_value="lib_result")
        raw_lib = FakeLibraryInstance({"new_page": method})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            result = mgr._execute_any_keyword_generic("New Page", ["url"], ns)

        assert result == "lib_result"
        method.assert_called_once_with("url")

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

    def test_library_method_raising_exception_falls_to_builtin(self):
        """Exception in library method is caught; resolution falls to BuiltIn."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        def boom(*args):
            raise ValueError("element not found")

        raw_lib = FakeLibraryInstance({"click": boom})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MagicMock(spec=[])  # No attributes — no direct method fallback
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
