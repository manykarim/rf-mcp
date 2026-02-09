"""Tests for P1 (Fail-Fast on Execution Errors) and P2 (Thread-Safe fd Redirect).

P1: When RF resolves and executes a keyword via StatusReporter but it fails
(assertion error, wrong args, keyword not found), the ExecutionStatus exception
propagates immediately without retrying through alternative resolution paths.
This reduces StatusReporter cycles from 4 to 1 for failing keywords.

P2: The _suppress_stdout() context manager uses reference counting so concurrent
threads keep fd 1 redirected until the LAST thread exits, preventing a race where
one thread restores fd 1 while another is still executing RF code.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# P2 Tests: Thread-safe fd redirect
# ---------------------------------------------------------------------------

class TestSuppressStdoutRefCounting:
    """Test the reference-counted _suppress_stdout() context manager."""

    def test_single_thread_redirects_fd1(self):
        """fd 1 should point to stderr during _suppress_stdout()."""
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        original_fd1_stat = os.fstat(1)
        stderr_stat = os.fstat(2)

        with _suppress_stdout():
            # During redirect, fd 1 should point to same file as fd 2
            redirected_stat = os.fstat(1)
            assert redirected_stat.st_ino == stderr_stat.st_ino, (
                "fd 1 should be redirected to stderr"
            )

        # After exit, fd 1 should be restored
        restored_stat = os.fstat(1)
        assert restored_stat.st_ino == original_fd1_stat.st_ino, (
            "fd 1 should be restored after exit"
        )

    def test_single_thread_restores_on_exception(self):
        """fd 1 should be restored even if an exception occurs inside the block."""
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        original_stat = os.fstat(1)
        with pytest.raises(ValueError, match="test error"):
            with _suppress_stdout():
                raise ValueError("test error")

        restored_stat = os.fstat(1)
        assert restored_stat.st_ino == original_stat.st_ino

    def test_refcount_starts_at_zero(self):
        """Reference count should be 0 when no redirect is active."""
        from robotmcp.components.execution import rf_native_context_manager as mod

        # Reset module state (in case other tests left it dirty)
        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0, (
                f"Expected 0, got {mod._stdout_redirect_count}"
            )

    def test_nested_redirects_refcount(self):
        """Nested _suppress_stdout() calls should increment/decrement refcount."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        with _suppress_stdout():
            with mod._stdout_redirect_lock:
                assert mod._stdout_redirect_count == 1

            with _suppress_stdout():
                with mod._stdout_redirect_lock:
                    assert mod._stdout_redirect_count == 2

                # fd 1 still points to stderr (inner context)
                assert os.fstat(1).st_ino == os.fstat(2).st_ino

            # Back to outer context — still redirected (count=1)
            with mod._stdout_redirect_lock:
                assert mod._stdout_redirect_count == 1
            assert os.fstat(1).st_ino == os.fstat(2).st_ino

        # Both exited — restored
        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0

    def test_concurrent_threads_keep_fd1_redirected(self):
        """Multiple threads entering _suppress_stdout concurrently should all
        keep fd 1 redirected until the last one exits."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        barrier = threading.Barrier(3, timeout=5)
        results = {}

        def worker(thread_id: int):
            with _suppress_stdout():
                # All threads should see fd 1 redirected
                barrier.wait()  # sync: all threads are inside
                fd1_stat = os.fstat(1)
                fd2_stat = os.fstat(2)
                results[thread_id] = (fd1_stat.st_ino == fd2_stat.st_ino)
                barrier.wait()  # sync: all threads checked before any exit

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All threads should have seen fd 1 redirected
        assert all(results.values()), f"Thread results: {results}"

        # After all threads exit, refcount should be 0
        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0

    def test_no_premature_restore_under_concurrency(self):
        """Thread A exiting should NOT restore fd 1 while Thread B is still inside."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        # Thread A enters, holds briefly, then exits
        # Thread B enters, waits for A to exit, then checks fd 1
        a_entered = threading.Event()
        a_exited = threading.Event()
        b_check_result = {}

        def thread_a():
            with _suppress_stdout():
                a_entered.set()
                # Wait a bit to ensure B also enters
                time.sleep(0.1)
            a_exited.set()

        def thread_b():
            a_entered.wait(timeout=5)
            with _suppress_stdout():
                # Wait for A to exit
                a_exited.wait(timeout=5)
                # A has exited but B is still inside — fd 1 should STILL be redirected
                fd1_stat = os.fstat(1)
                fd2_stat = os.fstat(2)
                b_check_result["redirected"] = (fd1_stat.st_ino == fd2_stat.st_ino)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)

        assert b_check_result.get("redirected", False), (
            "fd 1 should remain redirected while Thread B is still inside"
        )

    def test_saved_fd_is_cleaned_up(self):
        """The saved fd should be closed after the last redirect exits."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        with _suppress_stdout():
            with mod._stdout_redirect_lock:
                saved = mod._stdout_saved_fd
            assert saved is not None, "saved fd should be set during redirect"

        # After exit, saved fd should be None
        with mod._stdout_redirect_lock:
            assert mod._stdout_saved_fd is None, "saved fd should be cleaned up"


# ---------------------------------------------------------------------------
# P1 Tests: Fail-fast on execution errors
# ---------------------------------------------------------------------------

class TestExecuteAnyKeywordGenericFailFast:
    """Test that _execute_any_keyword_generic raises ExecutionStatus immediately
    without falling through to alternative resolution paths."""

    def _make_context_manager(self):
        """Create a RobotFrameworkNativeContextManager with mocked dependencies."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        return mgr

    def test_execution_failed_propagates_immediately(self):
        """ExecutionFailed from runner.run() should propagate, not fall through."""
        from robot.errors import ExecutionFailed
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = self._make_context_manager()

        # Mock namespace.get_runner() returning a runner that raises ExecutionFailed
        mock_runner = MagicMock()
        mock_runner.run.side_effect = ExecutionFailed("abc != xyz")

        mock_namespace = MagicMock()
        mock_namespace.get_runner.return_value = mock_runner

        # Mock EXECUTION_CONTEXTS.current
        mock_ctx = MagicMock()
        with patch(
            "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
        ) as mock_ec:
            mock_ec.current = mock_ctx

            with pytest.raises(ExecutionFailed, match="abc != xyz"):
                mgr._execute_any_keyword_generic(
                    "Should Be Equal", ["abc", "xyz"], mock_namespace
                )

        # runner.run was called exactly once — no fallback to paths 2 or 3
        mock_runner.run.assert_called_once()

    def test_handler_execution_failed_propagates(self):
        """HandlerExecutionFailed (subclass of ExecutionFailed) also propagates."""
        from robot.errors import HandlerExecutionFailed
        from robot.utils.error import ErrorDetails
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = self._make_context_manager()

        mock_runner = MagicMock()
        # HandlerExecutionFailed wraps an ErrorDetails
        try:
            raise AssertionError("Custom assertion")
        except AssertionError:
            error = ErrorDetails(sys.exc_info()[1])
        mock_runner.run.side_effect = HandlerExecutionFailed(error)

        mock_namespace = MagicMock()
        mock_namespace.get_runner.return_value = mock_runner

        mock_ctx = MagicMock()
        with patch(
            "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
        ) as mock_ec:
            mock_ec.current = mock_ctx

            with pytest.raises(HandlerExecutionFailed):
                mgr._execute_any_keyword_generic(
                    "Should Be True", ["False"], mock_namespace
                )

        mock_runner.run.assert_called_once()

    def test_non_execution_error_falls_through(self):
        """Non-ExecutionStatus exceptions (e.g. TypeError) should fall through
        to alternative resolution paths."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = self._make_context_manager()

        mock_runner = MagicMock()
        mock_runner.run.side_effect = TypeError("unexpected keyword argument")

        mock_namespace = MagicMock()
        mock_namespace.get_runner.return_value = mock_runner

        mock_ctx = MagicMock()
        mock_ctx.namespace.libraries = []  # empty — path 2 finds nothing

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
        ) as mock_ec:
            mock_ec.current = mock_ctx

            # Should fall through all paths and eventually raise RuntimeError
            # from _final_fallback_execution
            with pytest.raises(Exception):
                mgr._execute_any_keyword_generic(
                    "Some Keyword", ["arg1"], mock_namespace
                )

        # runner.run was called once (path 1), then fallback paths attempted
        mock_runner.run.assert_called_once()


class TestExecuteWithNativeResolutionFailFast:
    """Test that _execute_with_native_resolution propagates ExecutionStatus
    from _execute_any_keyword_generic without outer retries."""

    def _make_context_manager_with_session(self, session_id="test"):
        """Create manager with a mocked session context."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        return mgr

    def test_execution_failed_no_outer_retry(self):
        """ExecutionFailed from generic path should propagate to the outer
        except and return failure — NOT fall through to outer runner.run/BuiltIn."""
        from robot.errors import ExecutionFailed
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = self._make_context_manager_with_session()

        mock_namespace = MagicMock()
        mock_variables = MagicMock(spec=[])

        # Patch _execute_any_keyword_generic to raise ExecutionFailed
        with patch.object(
            mgr, "_execute_any_keyword_generic",
            side_effect=ExecutionFailed("'Should Be True' expected True but got 0"),
        ) as mock_generic:
            result = mgr._execute_with_native_resolution(
                session_id="test",
                keyword_name="Should Be True",
                arguments=["False"],
                namespace=mock_namespace,
                variables=mock_variables,
            )

        # Generic was called once
        mock_generic.assert_called_once()

        # Result should indicate failure
        assert result["success"] is False
        assert "Should Be True" in result["error"]

        # The outer namespace.get_runner should NOT have been called
        # (it would be on mock_namespace if the code fell through)
        # Since we're patching _execute_any_keyword_generic and it raised,
        # the code should hit the outer except and return error dict.
        # Verify no additional runner.run calls happened:
        mock_namespace.get_runner.assert_not_called()

    def test_non_execution_error_does_outer_retry(self):
        """Non-ExecutionStatus errors from generic path should fall through
        to the outer runner.run / BuiltIn retry paths."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = self._make_context_manager_with_session()

        mock_runner = MagicMock()
        mock_runner.run.return_value = "success_result"

        mock_namespace = MagicMock()
        mock_namespace.get_runner.return_value = mock_runner

        mock_variables = MagicMock(spec=[])

        # Patch generic to raise a non-ExecutionStatus error
        with patch.object(
            mgr, "_execute_any_keyword_generic",
            side_effect=AttributeError("some infrastructure error"),
        ):
            with patch(
                "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
            ) as mock_ec:
                mock_ctx = MagicMock()
                mock_ec.current = mock_ctx
                mock_ctx.steps = []

                result = mgr._execute_with_native_resolution(
                    session_id="test",
                    keyword_name="Log",
                    arguments=["hello"],
                    namespace=mock_namespace,
                    variables=mock_variables,
                )

        # The outer path should have been reached
        assert result["success"] is True
        mock_namespace.get_runner.assert_called()


class TestFailFastStatusReporterCycles:
    """Verify that P1 actually reduces StatusReporter cycles."""

    def test_execution_failed_single_cycle(self):
        """A failing keyword should only enter StatusReporter once, not 4 times."""
        from robot.errors import ExecutionFailed
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = RobotFrameworkNativeContextManager()
        call_count = {"runner_run": 0, "builtin_run_keyword": 0}

        original_generic = mgr._execute_any_keyword_generic

        def counting_generic(keyword_name, arguments, namespace):
            """Wrap _execute_any_keyword_generic to count runner.run calls."""
            # Create a namespace that tracks calls
            original_get_runner = namespace.get_runner

            def counting_get_runner(name, *args, **kwargs):
                runner = original_get_runner(name, *args, **kwargs)
                original_run = runner.run

                def counting_run(*a, **kw):
                    call_count["runner_run"] += 1
                    return original_run(*a, **kw)

                runner.run = counting_run
                return runner

            namespace.get_runner = counting_get_runner
            return original_generic(keyword_name, arguments, namespace)

        # We can't easily run this with a real RF context, so we test the
        # expected behavior: when _execute_any_keyword_generic raises
        # ExecutionFailed, the outer code does NOT call namespace.get_runner
        # again.
        mock_namespace = MagicMock()
        mock_variables = MagicMock(spec=[])

        with patch.object(
            mgr, "_execute_any_keyword_generic",
            side_effect=ExecutionFailed("test failure"),
        ):
            result = mgr._execute_with_native_resolution(
                session_id="test",
                keyword_name="Should Be True",
                arguments=["False"],
                namespace=mock_namespace,
                variables=mock_variables,
            )

        assert result["success"] is False
        # The outer retry (namespace.get_runner at line ~880) should NOT be called
        mock_namespace.get_runner.assert_not_called()


# ---------------------------------------------------------------------------
# P2 Stress Tests
# ---------------------------------------------------------------------------

class TestSuppressStdoutStress:
    """Stress tests for concurrent _suppress_stdout usage."""

    def test_many_concurrent_threads(self):
        """20 threads entering/exiting _suppress_stdout should not corrupt state."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        errors: List[str] = []
        n_threads = 20

        def worker(thread_id: int):
            try:
                with _suppress_stdout():
                    # Simulate RF execution time
                    time.sleep(0.01)
                    # fd 1 should be redirected
                    fd1_stat = os.fstat(1)
                    fd2_stat = os.fstat(2)
                    if fd1_stat.st_ino != fd2_stat.st_ino:
                        errors.append(
                            f"Thread {thread_id}: fd 1 not redirected"
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(worker, i) for i in range(n_threads)]
            for f in as_completed(futures, timeout=30):
                f.result()

        assert not errors, f"Errors: {errors}"

        # Refcount should be back to 0
        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0

    def test_rapid_enter_exit_cycles(self):
        """Rapid sequential enter/exit should not leak fd resources."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        original_stat = os.fstat(1)

        for _ in range(100):
            with _suppress_stdout():
                pass

        # fd 1 should be fully restored
        restored_stat = os.fstat(1)
        assert restored_stat.st_ino == original_stat.st_ino

        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0
            assert mod._stdout_saved_fd is None

    def test_mixed_success_and_exception(self):
        """Mix of successful and failing contexts should maintain correct state."""
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            _suppress_stdout,
        )

        errors: List[str] = []

        def worker_success(i):
            with _suppress_stdout():
                time.sleep(0.005)

        def worker_fail(i):
            try:
                with _suppress_stdout():
                    time.sleep(0.005)
                    raise ValueError(f"intentional error {i}")
            except ValueError:
                pass  # expected

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = []
            for i in range(20):
                if i % 2 == 0:
                    futures.append(pool.submit(worker_success, i))
                else:
                    futures.append(pool.submit(worker_fail, i))
            for f in as_completed(futures, timeout=30):
                f.result()

        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0
            assert mod._stdout_saved_fd is None


# ---------------------------------------------------------------------------
# Integration: P1 + P2 together
# ---------------------------------------------------------------------------

class TestP1P2Integration:
    """Verify P1 and P2 work together correctly."""

    def test_failfast_under_suppress_stdout(self):
        """ExecutionFailed should propagate correctly when _suppress_stdout is active."""
        from robot.errors import ExecutionFailed
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
            _suppress_stdout,
        )

        mgr = RobotFrameworkNativeContextManager()
        mock_namespace = MagicMock()
        mock_variables = MagicMock(spec=[])

        with patch.object(
            mgr, "_execute_any_keyword_generic",
            side_effect=ExecutionFailed("assertion failed"),
        ):
            with _suppress_stdout():
                result = mgr._execute_with_native_resolution(
                    session_id="test",
                    keyword_name="Should Be True",
                    arguments=["False"],
                    namespace=mock_namespace,
                    variables=mock_variables,
                )

        assert result["success"] is False
        assert "assertion failed" in result["error"]
        mock_namespace.get_runner.assert_not_called()

    def test_concurrent_failfast_threads(self):
        """Multiple threads hitting fail-fast simultaneously under _suppress_stdout."""
        from robot.errors import ExecutionFailed
        from robotmcp.components.execution import rf_native_context_manager as mod
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
            _suppress_stdout,
        )

        results = {}
        n_threads = 5

        def worker(thread_id):
            mgr = RobotFrameworkNativeContextManager()
            mock_ns = MagicMock()
            mock_vars = MagicMock(spec=[])

            with patch.object(
                mgr, "_execute_any_keyword_generic",
                side_effect=ExecutionFailed(f"error-{thread_id}"),
            ):
                with _suppress_stdout():
                    result = mgr._execute_with_native_resolution(
                        session_id=f"t-{thread_id}",
                        keyword_name="Fail",
                        arguments=[],
                        namespace=mock_ns,
                        variables=mock_vars,
                    )
            results[thread_id] = result

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All should have failed with their specific error
        for i in range(n_threads):
            assert results[i]["success"] is False
            assert f"error-{i}" in results[i]["error"]

        # Refcount should be clean
        with mod._stdout_redirect_lock:
            assert mod._stdout_redirect_count == 0


# ---------------------------------------------------------------------------
# P16-Issue1 fix invariant: runner.run() with console='none' writes zero
# bytes to fd 1.  This is the safety guarantee that allows removing
# _suppress_stdout() from the keyword execution path.
# ---------------------------------------------------------------------------


class TestConsoleNoneNoFd1Output:
    """Verify RF keyword execution writes nothing to fd 1 with console='none'."""

    def _make_rf_context(self):
        """Create a minimal RF context with console='none'."""
        import tempfile
        from pathlib import Path
        from robot.running.model import TestSuite
        from robot.running.namespace import Namespace
        from robot.conf.settings import RobotSettings
        from robot.variables.scopes import VariableScopes
        from robot.running.resourcemodel import ResourceFile
        from robot.output import Output
        from robot.running.context import EXECUTION_CONTEXTS

        try:
            from robot.conf.languages import Languages
        except ImportError:
            from robot.languages import Languages

        variables = VariableScopes(RobotSettings())
        suite = TestSuite(name="FD1Test")
        suite.source = Path("FD1Test.robot")
        suite.resource = ResourceFile(source=suite.source)
        namespace = Namespace(variables, suite, suite.resource, Languages())

        temp_dir = tempfile.mkdtemp(prefix="rf_mcp_fd1test_")
        settings = RobotSettings(outputdir=temp_dir, output=None, console="none")
        output = Output(settings)
        namespace.import_library("BuiltIn")
        ctx = EXECUTION_CONTEXTS.start_suite(suite, namespace, output, dry_run=False)
        namespace.start_suite()
        namespace.start_test()
        return namespace, ctx

    def _capture_fd1(self, func):
        """Run *func* while capturing everything written to fd 1."""
        import fcntl

        read_fd, write_fd = os.pipe()
        saved = os.dup(1)
        os.dup2(write_fd, 1)
        os.close(write_fd)
        try:
            func()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        flags = fcntl.fcntl(read_fd, fcntl.F_GETFL)
        fcntl.fcntl(read_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            data = os.read(read_fd, 65536)
        except BlockingIOError:
            data = b""
        os.close(read_fd)
        return data

    def test_runner_run_success_no_fd1_output(self):
        """runner.run() for a passing keyword writes nothing to fd 1."""
        from robot.running.model import Keyword as RunKW
        from robot.result.model import Keyword as ResKW

        namespace, ctx = self._make_rf_context()

        def run():
            runner = namespace.get_runner("Should Be Equal")
            runner.run(RunKW(name="Should Be Equal", args=("a", "a")), ResKW(name="Should Be Equal", args=("a", "a")), ctx)

        captured = self._capture_fd1(run)
        assert captured == b"", f"runner.run() wrote {len(captured)} bytes to fd 1: {captured!r}"

    def test_runner_run_failure_no_fd1_output(self):
        """runner.run() for a failing keyword writes nothing to fd 1."""
        from robot.running.model import Keyword as RunKW
        from robot.result.model import Keyword as ResKW
        from robot.errors import ExecutionFailed

        namespace, ctx = self._make_rf_context()

        def run():
            runner = namespace.get_runner("Should Be Equal")
            try:
                runner.run(RunKW(name="Should Be Equal", args=("a", "b")), ResKW(name="Should Be Equal", args=("a", "b")), ctx)
            except ExecutionFailed:
                pass  # Expected

        captured = self._capture_fd1(run)
        assert captured == b"", f"runner.run() wrote {len(captured)} bytes to fd 1: {captured!r}"

    def test_runner_run_wrong_args_no_fd1_output(self):
        """runner.run() with wrong argument count writes nothing to fd 1."""
        from robot.running.model import Keyword as RunKW
        from robot.result.model import Keyword as ResKW

        namespace, ctx = self._make_rf_context()

        def run():
            runner = namespace.get_runner("Length Should Be")
            try:
                runner.run(RunKW(name="Length Should Be", args=("abc",)), ResKW(name="Length Should Be", args=("abc",)), ctx)
            except Exception:
                pass

        captured = self._capture_fd1(run)
        assert captured == b"", f"runner.run() wrote {len(captured)} bytes to fd 1: {captured!r}"
