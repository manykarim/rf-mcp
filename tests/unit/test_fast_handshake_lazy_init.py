"""Lazy MCP-handshake init (change: fast-mcp-handshake-lazy-init).

The MCP handshake was blocked ~8s (cold) behind constructing the
ExecutionCoordinator at import, which preloaded libdoc for 17 RF libraries.
Both are now deferred: the doc storage populates lazily behind properties, and
the execution_engine global is a transparent proxy materialized by a warm-up
thread / first tool call.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from robotmcp.utils.rf_libdoc_integration import RobotFrameworkDocStorage


class TestLazyLibdocStorage:
    def test_construction_does_not_populate(self):
        s = RobotFrameworkDocStorage()
        assert s._initialized is False
        assert s._libraries == {}

    def test_property_access_triggers_population(self):
        s = RobotFrameworkDocStorage()
        assert len(s.libraries) > 0
        assert s._initialized is True

    def test_find_spec_gate_skips_uninstalled_keeps_stdlib_and_platynui(self):
        s = RobotFrameworkDocStorage()
        libs = s.libraries
        fi = s.failed_imports
        assert "BuiltIn" in libs  # stdlib (robot.libraries) — never gated
        if "DatabaseLibrary" in s.common_libraries:
            # Either loaded (installed via the `database` extra, e.g. --all-extras
            # CI) or cleanly gated out; never a hard crash. Do not assume absent.
            assert "DatabaseLibrary" in libs or fi.get("DatabaseLibrary") == "not installed (skipped)"
        if "PlatynUI.BareMetal" in s.common_libraries:
            # gated by module name "PlatynUI", not dist "robotframework-PlatynUI"
            assert "PlatynUI.BareMetal" in libs or "PlatynUI.BareMetal" in fi

    def test_concurrent_first_access_populates_once(self):
        s = RobotFrameworkDocStorage()
        calls = []
        orig = s._initialize_libraries

        def counting():
            calls.append(1)
            orig()

        s._initialize_libraries = counting
        threads = [threading.Thread(target=lambda: s.libraries) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1  # double-checked lock → exactly one population


class TestLazyEngineProxy:
    def test_import_constructs_no_coordinator_or_libdoc(self):
        code = (
            "import robotmcp.server as s;"
            "assert s._real_execution_engine is None, 'coordinator built at import';"
            "assert type(s.execution_engine).__name__ == '_LazyEngineProxy';"
            "from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage;"
            "assert get_rf_doc_storage()._initialized is False, 'libdoc populated at import';"
            "print('OK')"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]

    def test_factory_is_idempotent_and_applies_serialization(self):
        import robotmcp.server as s

        eng1 = s._get_execution_engine()
        eng2 = s._get_execution_engine()
        assert eng1 is eng2  # published once
        assert eng1.keyword_executor is not None  # enhanced serialization target present

    def test_mock_patch_substitutes_what_handlers_see(self):
        from unittest import mock
        import robotmcp.server as s

        with mock.patch("robotmcp.server.execution_engine", "SENTINEL"):
            assert s.execution_engine == "SENTINEL"

    def test_lazy_init_zero_restores_eager(self):
        code = (
            "import robotmcp.server as s;"
            "assert s._real_execution_engine is not None, 'not eager under LAZY_INIT=0';"
            "print('OK')"
        )
        env = dict(os.environ, ROBOTMCP_LAZY_INIT="0")
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
