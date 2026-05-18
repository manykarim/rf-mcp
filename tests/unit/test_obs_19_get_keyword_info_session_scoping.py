"""OBS-19 — get_keyword_info(mode="keyword") must honour session_id.

The session_id parameter was accepted at the API surface and
advertised in the docstring but never consulted by the keyword/global
branch. Direct repro:

  get_keyword_info(mode="keyword", keyword_name="Click",
                   session_id="dummy-nonexistent-session")
  → success=True, returns Browser.Click documentation

This was the same class of silent-parameter-ignore defect as the
original OBS library_name bug we fixed in PR #70.

The fix:
1. Resolve allowed_libraries from session.imported_libraries (+ neutral
   helpers) when session_id is provided without library_name.
2. Pass allowed_libraries into execution_engine.get_keyword_documentation.
3. Filter matches[] to the allowed set in the LibDoc path.
4. When the keyword exists only in non-allowed libraries, return a
   library-mismatch error + plugin's alternative hint via the existing
   shared API at LibraryPluginManager.validate_keyword_for_session.

These tests pin all four pieces.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import get_keyword_info


def _fn(tool):
    return getattr(tool, "fn", tool)


def _make_kw_record(name: str, library: str):
    """Synthetic RFKeywordInfo-shaped object for libdoc returns."""
    kw = MagicMock()
    kw.name = name
    kw.library = library
    kw.args = ["arg1"]
    kw.arg_types = ["str"]
    kw.doc = f"{name} doc"
    kw.short_doc = f"{name} short"
    kw.tags = []
    kw.is_deprecated = False
    kw.source = ""
    kw.lineno = 0
    return kw


@pytest.fixture
def patched_engines():
    """Stub session manager + libdoc storage so tests don't need RF runtime."""
    sess_mgr = MagicMock()
    storage = MagicMock()
    storage.is_available = MagicMock(return_value=True)
    storage.get_keywords_documentation_all = MagicMock(return_value=[])
    storage.get_keyword_documentation = MagicMock(return_value=None)
    engine = MagicMock()
    engine.session_manager = sess_mgr
    engine.get_keyword_documentation = MagicMock(return_value={
        "success": False, "error": "default stub",
    })
    with patch("robotmcp.server.execution_engine", engine):
        yield {
            "sess_mgr": sess_mgr,
            "storage": storage,
            "engine": engine,
        }


@pytest.mark.asyncio
class TestSessionScopedLookup:
    """When session_id is provided without library_name, the lookup
    scopes to the session's imported libraries (plus neutral helpers)."""

    async def test_session_id_passes_allowed_libraries_to_engine(
        self, patched_engines,
    ):
        sess = MagicMock()
        sess.imported_libraries = ["Browser", "BuiltIn"]
        sess.explicit_library_preference = "Browser"
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)
        patched_engines["engine"].get_keyword_documentation = MagicMock(
            return_value={"success": True, "matches": []}
        )
        await _fn(get_keyword_info)(
            mode="keyword", keyword_name="Click", session_id="sess-x",
        )
        # The engine should receive allowed_libraries scoped by the
        # session (Browser+BuiltIn) plus neutral helpers (Collections,
        # String, DateTime, OperatingSystem, Process, XML).
        _, kwargs = patched_engines["engine"].get_keyword_documentation.call_args
        allowed = kwargs.get("allowed_libraries")
        assert allowed is not None
        assert "Browser" in allowed
        assert "BuiltIn" in allowed
        # Neutral helpers preserved.
        assert "Collections" in allowed
        assert "String" in allowed

    async def test_keyword_in_allowed_lib_returns_match(self, patched_engines):
        """Click is in Browser; session imports Browser → success."""
        sess = MagicMock()
        sess.imported_libraries = ["Browser"]
        sess.explicit_library_preference = "Browser"
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)
        patched_engines["engine"].get_keyword_documentation = MagicMock(
            return_value={
                "success": True,
                "matches": [{"name": "Click", "library": "Browser"}],
            }
        )
        result = await _fn(get_keyword_info)(
            mode="keyword", keyword_name="Click", session_id="sess-x",
        )
        assert result["success"] is True
        assert "matches" in result

    async def test_keyword_not_in_allowed_lib_returns_error(
        self, patched_engines,
    ):
        """Click Element is in SL; session imports only Browser →
        error with found_in_other_libraries indicator."""
        sess = MagicMock()
        sess.imported_libraries = ["Browser"]
        sess.explicit_library_preference = "Browser"
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)
        # Simulate the engine returning the OBS-19 enriched not-found
        # response.
        patched_engines["engine"].get_keyword_documentation = MagicMock(
            return_value={
                "success": False,
                "error": (
                    "Keyword 'Click Element' is not available in this "
                    "session's libraries (...). Found in: ['SeleniumLibrary']."
                ),
                "found_in_other_libraries": ["SeleniumLibrary"],
            }
        )
        result = await _fn(get_keyword_info)(
            mode="keyword", keyword_name="Click Element", session_id="sess-x",
        )
        assert result["success"] is False
        assert "found_in_other_libraries" in result
        # Plugin hint enrichment fires.
        assert "hint" in result


@pytest.mark.asyncio
class TestNonexistentSession:
    """When session_id refers to a session that doesn't exist, return
    a clear error rather than falling through to global lookup."""

    async def test_unknown_session_id_returns_error(self, patched_engines):
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=None)
        result = await _fn(get_keyword_info)(
            mode="keyword", keyword_name="Click",
            session_id="dummy-nonexistent",
        )
        assert result["success"] is False
        assert "dummy-nonexistent" in result["error"]
        assert "not found" in result["error"].lower()


@pytest.mark.asyncio
class TestBackwardsCompat:
    """Behaviour without session_id (the existing API) is unchanged."""

    async def test_no_session_id_uses_global_lookup(self, patched_engines):
        # No session_manager call should happen.
        patched_engines["engine"].get_keyword_documentation = MagicMock(
            return_value={"success": True, "matches": [
                {"name": "Click", "library": "Browser"},
                {"name": "Click", "library": "SeleniumLibrary"},
            ]}
        )
        result = await _fn(get_keyword_info)(
            mode="keyword", keyword_name="Click",
        )
        assert result["success"] is True
        # Both libraries present (no filter).
        libs = {m["library"] for m in result["matches"]}
        assert libs == {"Browser", "SeleniumLibrary"}
        # session_manager.get_session was NOT consulted.
        patched_engines["sess_mgr"].get_session.assert_not_called()

    async def test_library_name_supersedes_session_id(self, patched_engines):
        """When BOTH library_name and session_id are provided,
        library_name takes precedence (explicit per-call scope)."""
        sess = MagicMock()
        sess.imported_libraries = ["Browser"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)
        patched_engines["engine"].get_keyword_documentation = MagicMock(
            return_value={"success": True, "keyword": {"name": "Click"}}
        )
        await _fn(get_keyword_info)(
            mode="keyword", keyword_name="Click",
            library_name="SeleniumLibrary",  # explicit
            session_id="sess-x",  # session imports only Browser
        )
        # When library_name is explicit, the resolver should NOT
        # derive allowed_libraries from the session — the engine
        # receives the explicit library_name + allowed_libraries=None.
        _, kwargs = patched_engines["engine"].get_keyword_documentation.call_args
        args = patched_engines["engine"].get_keyword_documentation.call_args.args
        assert args[1] == "SeleniumLibrary"  # library_name (positional)
        assert kwargs.get("allowed_libraries") is None


class TestEngineFilter:
    """The execution_coordinator.get_keyword_documentation filter logic
    itself — independent of the server wrapper."""

    def test_libdoc_path_filters_to_allowed_libraries(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=True)
        coord.rf_doc_storage.get_keywords_documentation_all = MagicMock(
            return_value=[
                _make_kw_record("Go To", "Browser"),
                _make_kw_record("Go To", "SeleniumLibrary"),
            ]
        )
        # Allowed = Browser only → SL excluded.
        result = coord.get_keyword_documentation(
            "Go To", allowed_libraries=["Browser", "BuiltIn"],
        )
        assert result["success"] is True
        libs = {m["library"] for m in result["matches"]}
        assert libs == {"Browser"}

    def test_libdoc_path_reports_other_libraries_when_no_match(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=True)
        coord.rf_doc_storage.get_keywords_documentation_all = MagicMock(
            return_value=[
                _make_kw_record("Click Element", "SeleniumLibrary"),
            ]
        )
        # Keyword exists in SL but session allows only Browser.
        result = coord.get_keyword_documentation(
            "Click Element", allowed_libraries=["Browser", "BuiltIn"],
        )
        assert result["success"] is False
        assert "SeleniumLibrary" in str(result.get("found_in_other_libraries"))

    def test_libdoc_path_unchanged_when_allowed_is_none(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=True)
        coord.rf_doc_storage.get_keywords_documentation_all = MagicMock(
            return_value=[
                _make_kw_record("Go To", "Browser"),
                _make_kw_record("Go To", "SeleniumLibrary"),
            ]
        )
        # allowed_libraries=None → both libraries kept.
        result = coord.get_keyword_documentation("Go To")
        libs = {m["library"] for m in result["matches"]}
        assert libs == {"Browser", "SeleniumLibrary"}
