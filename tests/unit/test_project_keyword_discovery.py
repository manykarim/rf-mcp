"""Regression tests for discovery of a project's own keywords (change:
project-keyword-discovery).

rf-mcp could EXECUTE a project's custom-library and resource-file keywords but not
FIND or DESCRIBE them: execution goes through RF's namespace, while find_keywords and
get_keyword_info read a LibDoc-backed store that held only rf-mcp's own libraries.
Measured: `find_keywords(query="Acme Add")` returned 336 results topped by
`Close Window`, and `get_keyword_info` said "not found in any loaded library", while
`execute_step` ran the keyword fine (evidence:
experiments/project_keywords_discovery_evidence.md).

Run: uv run pytest tests/unit/test_project_keyword_discovery.py -q
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage


# --- fixtures: a custom library and a resource file, as a real project has -----

@pytest.fixture
def acme_lib(tmp_path, monkeypatch):
    """A custom Python keyword library importable like one installed in a .venv."""
    pkg = tmp_path / "AcmeLib.py"
    pkg.write_text(textwrap.dedent('''
        class AcmeLib:
            """Acme domain keywords."""
            ROBOT_LIBRARY_SCOPE = "GLOBAL"

            def acme_add(self, a: int, b: int) -> int:
                """Add two integers and return the sum."""
                return int(a) + int(b)

            def acme_greet(self, who: str = "world") -> str:
                """Return a greeting for ``who``."""
                return f"hello {who}"
    ''').strip() + "\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "AcmeLib"


@pytest.fixture
def acme_resource(tmp_path):
    """A project resource file of user keywords with args, defaults and docs."""
    res = tmp_path / "acme_kw.resource"
    res.write_text(textwrap.dedent('''
        *** Keywords ***
        Acme Login
            [Documentation]    High-level user keyword.
            [Arguments]    ${user}    ${password}=secret
            Log    ${user} ${password}

        Acme Sum Should Be
            [Documentation]    Assert that a+b equals the expected value.
            [Arguments]    ${a}    ${b}    ${expected}
            Should Be Equal As Integers    ${a}    ${expected}
    ''').strip() + "\n", encoding="utf-8")
    return str(res)


@pytest.fixture
def storage():
    s = get_rf_doc_storage()
    yield s
    # Keep the process-global singleton clean between tests.
    for sid in list(getattr(s, "_session_sources", {})):
        s.forget_session_sources(sid)


# =============================================================================
# Registration
# =============================================================================


class TestRegistration:
    def test_custom_library_is_registered(self, storage, acme_lib):
        r = storage.register_project_source("s1", acme_lib)
        assert r["success"] is True
        assert r["type"] == "LIBRARY"
        assert r["keywords"] >= 2

    def test_resource_file_is_registered(self, storage, acme_resource):
        r = storage.register_project_source("s1", acme_resource)
        assert r["success"] is True
        assert r["type"] == "RESOURCE"
        assert r["keywords"] == 2

    def test_unparseable_source_degrades_without_raising(self, storage, tmp_path):
        """A parse failure must not break the import - execution still works."""
        bad = tmp_path / "broken.resource"
        bad.write_text("*** Keywords ***\n    [Arguments]\n", encoding="utf-8")
        r = storage.register_project_source("s1", str(tmp_path / "nope.resource"))
        assert r["success"] is False
        assert "error" in r

    def test_registration_is_cached_by_mtime(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        before = len(storage._source_cache())
        storage.register_project_source("s2", acme_resource)
        assert len(storage._source_cache()) == before, "same source should reuse LibDoc"

    def test_edited_resource_is_reread(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        p = Path(acme_resource)
        p.write_text(p.read_text(encoding="utf-8") + textwrap.dedent('''
        Acme Extra
            [Documentation]    Added later.
            Log    extra
        '''), encoding="utf-8")
        import os
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 10))
        r = storage.register_project_source("s1", acme_resource)
        assert r["keywords"] == 3


# =============================================================================
# Lookup - the defect itself
# =============================================================================


class TestLookup:
    def test_custom_library_keyword_is_findable(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        matches = storage.project_keyword_matches("Acme Add", "s1")
        assert [m.name for m in matches] == ["Acme Add"]

    def test_resource_keyword_is_findable(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        matches = storage.project_keyword_matches("Acme Login", "s1")
        assert [m.name for m in matches] == ["Acme Login"]

    def test_lookup_is_name_normalized(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        assert storage.project_keyword_matches("acme_add", "s1")
        assert storage.project_keyword_matches("ACME ADD", "s1")

    def test_unknown_keyword_returns_nothing(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        assert storage.project_keyword_matches("No Such Keyword", "s1") == []


# =============================================================================
# Session scoping - a correctness requirement, not polish
# =============================================================================


class TestSessionScoping:
    def test_one_session_does_not_see_another_s_project_keywords(
        self, storage, acme_resource
    ):
        """The store is a process-global singleton. Leaking project A's keywords into
        a session driving project B would make discovery advertise keywords that
        execution then cannot run."""
        storage.register_project_source("sA", acme_resource)
        assert storage.project_keyword_matches("Acme Login", "sA")
        assert storage.project_keyword_matches("Acme Login", "sB") == []
        assert storage.project_libraries("sB") == {}

    def test_no_session_id_yields_nothing(self, storage, acme_resource):
        storage.register_project_source("sA", acme_resource)
        assert storage.project_libraries(None) == {}
        assert storage.project_keyword_matches("Acme Login", None) == []

    def test_forget_session_sources(self, storage, acme_resource):
        storage.register_project_source("sA", acme_resource)
        storage.forget_session_sources("sA")
        assert storage.project_libraries("sA") == {}

    def test_bundled_keywords_unaffected(self, storage, acme_resource):
        storage.register_project_source("sA", acme_resource)
        assert storage.find_keyword("Log") is not None


# =============================================================================
# Filtering by source
# =============================================================================


class TestSourceFilter:
    def test_all_project_keywords_for_a_session(self, storage, acme_lib, acme_resource):
        storage.register_project_source("s1", acme_lib)
        storage.register_project_source("s1", acme_resource)
        names = {k.name for k in storage.project_keywords("s1")}
        assert {"Acme Add", "Acme Greet", "Acme Login", "Acme Sum Should Be"} <= names

    def test_filter_selects_one_source(self, storage, acme_lib, acme_resource):
        storage.register_project_source("s1", acme_lib)
        storage.register_project_source("s1", acme_resource)
        only_res = {k.name for k in storage.project_keywords("s1", library_name="acme_kw")}
        assert only_res == {"Acme Login", "Acme Sum Should Be"}
        only_lib = {k.name for k in storage.project_keywords("s1", library_name="AcmeLib")}
        assert "Acme Add" in only_lib and "Acme Login" not in only_lib


# =============================================================================
# Metadata fidelity - described must match what execution accepts
# =============================================================================


class TestMetadataFidelity:
    def test_resource_keyword_reports_arguments_and_default(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        kw = storage.project_keyword_matches("Acme Login", "s1")[0]
        joined = " ".join(kw.args)
        assert "user" in joined
        assert "secret" in joined, "the argument's default must be reported"

    def test_custom_library_keyword_reports_types(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        kw = storage.project_keyword_matches("Acme Add", "s1")[0]
        joined = " ".join(kw.args) + " " + " ".join(kw.arg_types or [])
        assert "int" in joined, "type hints must be reported"

    def test_documentation_is_carried_through(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        kw = storage.project_keyword_matches("Acme Sum Should Be", "s1")[0]
        assert "expected value" in kw.doc

    def test_source_is_reported(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        kw = storage.project_keyword_matches("Acme Login", "s1")[0]
        assert kw.library == "acme_kw"


# =============================================================================
# Query scoring - exact names must win
# =============================================================================


class TestQueryScoring:
    def _matches(self, query, session_id="s1", library_name=None):
        from robotmcp.server import _project_keyword_matches_for_query

        return _project_keyword_matches_for_query(query, session_id, library_name)

    def test_exact_name_scores_highest(self, storage, acme_lib):
        """`query="Acme Add"` used to return 336 results topped by `Close Window`."""
        storage.register_project_source("s1", acme_lib)
        m = self._matches("Acme Add")
        assert m and m[0]["keyword_name"] == "Acme Add"
        assert m[0]["confidence"] == 1.0

    def test_case_and_underscores_are_normalized(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        assert self._matches("acme_add")[0]["confidence"] == 1.0

    def test_unrelated_query_returns_nothing(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        assert self._matches("navigate to a web page and click") == []

    def test_no_session_returns_nothing(self, storage, acme_lib):
        storage.register_project_source("s1", acme_lib)
        assert self._matches("Acme Add", session_id=None) == []

    def test_results_carry_the_project_marker(self, storage, acme_resource):
        storage.register_project_source("s1", acme_resource)
        m = self._matches("Acme Login")
        assert m[0]["from_project"] is True
        assert m[0]["library"] == "acme_kw"


class TestMergeIntoDiscovery:
    def test_project_matches_lead_the_merged_list(self, storage, acme_lib):
        from robotmcp.server import _merge_project_keywords

        storage.register_project_source("s1", acme_lib)
        discovery = {"matches": [
            {"keyword_name": "Close Window", "library": "SeleniumLibrary", "confidence": 0.75},
        ]}
        _merge_project_keywords(discovery, "Acme Add", "s1")
        assert discovery["matches"][0]["keyword_name"] == "Acme Add"
        # Sibling keywords from the same project source may also surface (useful);
        # the guarantee is that the EXACT match leads.
        assert discovery["project_keyword_count"] >= 1
        # the bundled match is preserved, not dropped
        assert any(m["keyword_name"] == "Close Window" for m in discovery["matches"])

    def test_no_session_leaves_discovery_untouched(self, storage):
        from robotmcp.server import _merge_project_keywords

        discovery = {"matches": [{"keyword_name": "Close Window", "library": "SeleniumLibrary"}]}
        _merge_project_keywords(discovery, "Acme Add", None)
        assert "project_keyword_count" not in discovery
        assert len(discovery["matches"]) == 1

    def test_strict_library_scopes_to_the_named_source(self, storage, acme_resource):
        from robotmcp.server import _merge_project_keywords

        storage.register_project_source("s1", acme_resource)
        discovery = {"matches": [
            {"keyword_name": "Close Window", "library": "SeleniumLibrary", "confidence": 0.9},
        ]}
        _merge_project_keywords(discovery, "Acme Login", "s1",
                                library_name="acme_kw", strict_library=True)
        assert all(m["library"] == "acme_kw" for m in discovery["matches"])


class TestActionableMiss:
    """A miss used to say only "not found in any loaded library" - true, but it did
    not say that importing the defining source would fix it."""

    def _lookup(self, name, session_id=None):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )

        return ExecutionCoordinator().get_keyword_documentation(
            name, allowed_libraries=[], session_id=session_id
        )

    def test_miss_names_the_import_calls(self):
        r = self._lookup("Totally Unknown Keyword XYZ")
        assert r["success"] is False
        assert "import_library" in r["error"]
        assert "import_resource" in r["error"]

    def test_registered_project_keyword_is_found(self, storage, acme_resource):
        storage.register_project_source("sess-miss", acme_resource)
        r = self._lookup("Acme Login", session_id="sess-miss")
        assert r["success"] is True
        assert r["matches"][0]["library"] == "acme_kw"

    def test_unregistered_session_still_misses(self, storage, acme_resource):
        storage.register_project_source("sess-other", acme_resource)
        r = self._lookup("Acme Login", session_id="sess-none")
        assert r["success"] is False
