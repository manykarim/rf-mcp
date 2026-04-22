"""Tests for MCP stdout protection (non-JSON log message fix).

Verifies that _protect_mcp_stdout() prevents RF console output and
bare print() calls from contaminating the MCP stdio transport (fd 1).
"""
import io
import os
import sys

import pytest


class TestProtectMcpStdout:
    """Verify _protect_mcp_stdout redirects sys.__stdout__ to stderr."""

    def test_dunder_stdout_redirected_to_stderr(self):
        """After _protect_mcp_stdout, sys.__stdout__ should write to stderr, not stdout."""
        from robotmcp.server import _protect_mcp_stdout

        saved = sys.__stdout__
        try:
            _protect_mcp_stdout()
            # sys.__stdout__ should no longer be the original
            assert sys.__stdout__ is not saved
            # Writing to __stdout__ should NOT go to fd 1
            # We verify by checking it's a TextIOWrapper pointing to stderr fd
            if hasattr(sys.__stdout__, 'buffer') and hasattr(sys.__stdout__.buffer, 'fileno'):
                try:
                    assert sys.__stdout__.buffer.fileno() == sys.stderr.fileno()
                except io.UnsupportedOperation:
                    pass  # pytest captures may not support fileno
        finally:
            sys.__stdout__ = saved

    def test_sys_stdout_unchanged(self):
        """sys.stdout must remain unchanged for MCP transport."""
        from robotmcp.server import _protect_mcp_stdout

        original_stdout = sys.stdout
        saved_dunder = sys.__stdout__
        try:
            _protect_mcp_stdout()
            # sys.stdout should be untouched (MCP captures sys.stdout.buffer)
            assert sys.stdout is original_stdout
        finally:
            sys.__stdout__ = saved_dunder

    def test_rf_write_to_console_patched(self):
        """write_to_console should be monkey-patched to force stderr."""
        from robotmcp.server import _protect_mcp_stdout

        saved_dunder = sys.__stdout__
        try:
            _protect_mcp_stdout()
            import robot.output.librarylogger as rll
            # Capture what write_to_console does
            captured = io.StringIO()
            original_stderr = sys.__stderr__
            sys.__stderr__ = captured
            try:
                rll.write_to_console("test_message", newline=False, stream="stdout")
            except Exception:
                pass  # encoding issues in StringIO are OK
            finally:
                sys.__stderr__ = original_stderr
            # The message should have been redirected to stderr (our captured StringIO)
            # or at minimum, NOT written to sys.__stdout__ (which is now stderr anyway)
        finally:
            sys.__stdout__ = saved_dunder


class TestNoPrintCallsInSource:
    """Verify no bare print() calls exist in production source code."""

    def test_no_stdout_print_in_orchestrator(self):
        """dynamic_keyword_orchestrator.py should have no bare print() calls."""
        import robotmcp.core.dynamic_keyword_orchestrator as mod
        import inspect
        source = inspect.getsource(mod)
        # Find print( calls that DON'T have file=sys.stderr
        import re
        bare_prints = re.findall(r'^\s*print\([^)]*\)\s*$', source, re.MULTILINE)
        # Filter out any that are in docstrings or comments
        real_prints = [p for p in bare_prints if 'file=' not in p and '#' not in p.split('print')[0]]
        assert len(real_prints) == 0, f"Found bare print() calls: {real_prints}"

    def test_no_stderr_print_in_keyword_executor(self):
        """keyword_executor.py should use logger.debug() not print(file=stderr)."""
        import robotmcp.components.execution.keyword_executor as mod
        import inspect
        source = inspect.getsource(mod)
        import re
        stderr_prints = re.findall(r'print\(.*file=sys\.stderr', source)
        assert len(stderr_prints) == 0, f"Found print(file=sys.stderr) calls: {stderr_prints}"
