"""build_test_suite safe persistence (change: build-suite-safe-persist).

The 2026-07-10 Docker capability experiment surfaced a corrupted generated
suite: a MiniMax agent wrote build_test_suite's ``rf_text`` to disk via the RF
``Create File`` keyword, which resolved ``${variables}`` and interpreted ``\\n``
escapes *inside the argument*, mangling the suite. build_test_suite itself
renders correctly (this file guards that); the real gap was the absence of a
safe persistence path. ``build_suite(output_path=...)`` now writes the suite via
plain file I/O, byte-for-byte, with no RF resolution.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.execution_coordinator import ExecutionCoordinator
from robotmcp.components.test_builder import TestBuilder


async def _fileproc_session(eng, sid, path):
    """Replay the exact file-proc pattern: multi-line Create File arg, a
    Get File assigned to ${file_content}, and a Should Contain that references
    the variable — the shape that was reported (mis-diagnosed) as F1."""
    sess = eng.session_manager.get_or_create_session(sid)
    sess.search_order = ["BuiltIn", "OperatingSystem"]
    sess.test_registry.start_test("File Processing Test")
    await eng.execute_step(
        "Create File", [path, "line1\nline2\nline3\n"], sid, use_context=True
    )
    await eng.execute_step(
        "Get File", [path], sid, use_context=True, assign_to="file_content"
    )
    await eng.execute_step(
        "Should Contain", ["${file_content}", "line1"], sid, use_context=True
    )
    await eng.execute_step("File Should Exist", [path], sid, use_context=True)
    sess.test_registry.end_test(status="pass")


@pytest.mark.asyncio
class TestBuildSuiteSafePersist:
    async def test_rf_text_is_correct_not_corrupted(self, tmp_path):
        # F1's *rendering* concern, directly refuted: build_suite escapes
        # newlines and PRESERVES ${assigned} references (does not resolve them).
        eng = ExecutionCoordinator()
        f = str(tmp_path / "report.txt")
        await _fileproc_session(eng, "persist-render", f)
        rf = (await TestBuilder(execution_engine=eng).build_suite(
            "persist-render", test_name=""
        ))["rf_text"]
        assert "line1\\nline2\\nline3\\n" in rf          # escaped, stays on one line
        assert "\n    Create File    " + f + "    line1\nline2" not in rf  # no raw break
        assert "${file_content} =" in rf                 # assignment kept as a var
        assert "Should Contain    ${file_content}" in rf  # arg kept as a var

    async def test_output_path_persists_byte_for_byte_and_parses(self, tmp_path):
        eng = ExecutionCoordinator()
        f = str(tmp_path / "report.txt")
        await _fileproc_session(eng, "persist-write", f)
        out = tmp_path / "suite.robot"
        res = await TestBuilder(execution_engine=eng).build_suite(
            "persist-write", test_name="", output_path=str(out)
        )
        assert res.get("output_error") is None
        assert res["output_path"] == str(out)
        assert res["output_bytes"] == len(res["rf_text"].encode("utf-8"))
        on_disk = out.read_text(encoding="utf-8")
        assert on_disk == res["rf_text"]                 # byte-for-byte, no resolution
        assert "${file_content} =" in on_disk            # var NOT resolved to 'line1…'
        assert "line1\\nline2\\nline3\\n" in on_disk      # escapes NOT expanded
        # The persisted suite must actually PARSE as Robot Framework.
        from robot.api import TestSuiteBuilder

        parsed = TestSuiteBuilder().build(str(out))
        assert parsed.tests, "persisted suite parsed but has no test cases"

    async def test_create_file_roundtrip_corrupts_documents_root_cause(self, tmp_path):
        # Documents WHY output_path exists: persisting suite text via the RF
        # Create File keyword resolves ${vars} and expands \n escapes → corruption.
        eng = ExecutionCoordinator()
        sid = "persist-corrupt"
        sess = eng.session_manager.get_or_create_session(sid)
        sess.search_order = ["BuiltIn", "OperatingSystem"]
        await eng.execute_step(
            "Set Variable", ["hello"], sid, use_context=True, assign_to="file_content"
        )
        bad = str(tmp_path / "bad.robot")
        await eng.execute_step(
            "Create File", [bad, "x ${file_content}\\ny"], sid, use_context=True
        )
        corrupted = open(bad, encoding="utf-8").read()
        assert "${file_content}" not in corrupted   # resolved away by RF
        assert "hello" in corrupted                  # to its runtime value
        assert "\n" in corrupted                     # \n escape expanded to a real newline
