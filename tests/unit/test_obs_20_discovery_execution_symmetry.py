"""OBS-20 — discovery filter mirrors Browser plugin's imported-libraries fallback.

Pre-fix: ``_filter_keywords_by_session_library`` only ran when
``session.explicit_library_preference`` was non-None. Meanwhile, the
Browser plugin's execute-time validation
(``browser_plugin.py:312-321``) ALSO falls back to imported libraries
when no explicit preference exists:

  if not pref:
      imported = getattr(session, "imported_libraries", []) or []
      if "Browser" not in imported:
          return None  # Browser not loaded — skip validation

Result: an agent in a session created with ``libraries=["Browser"]``
but no explicit preference set saw SL keywords in ``find_keywords``
results, picked one, and had it rejected at ``execute_step``. Same
tools, asymmetric filtering rules.

The SeleniumLibrary plugin at ``selenium_plugin.py:203-205`` is
explicitly asymmetric — it ONLY acts when
``explicit_library_preference.startswith("selenium")``. Does NOT use
imported-libraries fallback. OBS-20 mirrors this exact asymmetry —
it does NOT invent a clean XOR rule that would contradict production.

These tests pin the actual asymmetric rule:

1. Browser imported, no explicit preference → SL excluded from
   discovery (mirrors Browser plugin's execute rule).
2. SL imported, no explicit preference → Browser KEPT in discovery
   (mirrors SL plugin's lack of imported-fallback).
3. BOTH Browser + SL imported → SL still excluded (Browser plugin's
   rule fires when Browser is imported, regardless of SL also being
   imported).
4. Explicit preference set → unchanged behaviour (no regression on
   the existing path).
5. Neither Browser nor SL imported → no filter (existing behaviour).
6. ``library_name`` parameter takes precedence over session
   imported-fallback (existing precedence preserved).
7. Neutral libraries (BuiltIn, Collections, etc.) never excluded by
   this filter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _fn(tool):
    return getattr(tool, "fn", tool)


@pytest.fixture
def patched_engines():
    """Stub matcher/engine/externalisation so tests don't hit RF runtime."""
    discover_mock = AsyncMock()
    async def _ensure_loaded():
        return None
    sess_mgr = MagicMock()
    sess_mgr.get_session = MagicMock(return_value=None)
    engine = MagicMock()
    engine.session_manager = sess_mgr
    engine.search_keywords = MagicMock(return_value=[])
    engine.get_available_keywords = MagicMock(return_value=[])
    with patch(
        "robotmcp.server.keyword_matcher.discover_keywords", discover_mock,
    ), patch(
        "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
    ), patch(
        "robotmcp.server._externalize_response", side_effect=lambda *a: a[-1],
    ), patch(
        "robotmcp.server._track_tool_result", MagicMock(),
    ), patch(
        "robotmcp.server.execution_engine", engine,
    ):
        yield {"discover": discover_mock, "engine": engine, "sess_mgr": sess_mgr}


def _stub_semantic(matches):
    return {
        "success": True,
        "action_description": "test",
        "action_type": "click",
        "matches": matches,
        "total_matches": len(matches),
        "recommendations": [],
    }


@pytest.mark.asyncio
class TestImportedFallbackForBrowser:
    """OBS-20 core: when a session imports Browser but doesn't set
    explicit_library_preference, discovery still excludes SL."""

    async def test_browser_imported_no_preference_excludes_sl(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        sess.imported_libraries = ["Browser", "BuiltIn"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", session_id="s",
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # SL excluded by imported-fallback rule
        assert "SeleniumLibrary" not in libs, (
            f"OBS-20: Browser imported + no preference must exclude SL; "
            f"got libs={libs!r}"
        )
        assert "Browser" in libs
        # library_filter.source signals the imported-fallback path
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["source"] == "session_imported"
        assert lf["applied"] == "Browser"


@pytest.mark.asyncio
class TestSlAsymmetricRule:
    """OBS-20 honours SL plugin's lack of imported-fallback: SL
    imported + no preference does NOT exclude Browser."""

    async def test_sl_imported_no_preference_keeps_browser(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        # SL imported, Browser NOT imported
        sess.imported_libraries = ["SeleniumLibrary", "BuiltIn"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", session_id="s",
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # Both libraries' keywords remain — SL plugin has no
        # imported-fallback; mirror exactly.
        assert "Browser" in libs, (
            "OBS-20 asymmetry: SL imported without preference must NOT "
            "trigger an imported-fallback filter (SL plugin doesn't have "
            "one at selenium_plugin.py:203-205)"
        )
        # No library_filter applied
        assert "library_filter" not in result


@pytest.mark.asyncio
class TestMixedImports:
    """When BOTH Browser AND SL are imported (no explicit preference),
    Browser plugin's rule still fires — SL is excluded."""

    async def test_both_imported_no_preference_sl_still_excluded(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        sess.imported_libraries = ["Browser", "SeleniumLibrary", "BuiltIn"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", session_id="s",
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # SL excluded because Browser is imported (browser_plugin.py:312-321
        # rule). This is the production rule, regardless of SL also being
        # imported.
        assert "SeleniumLibrary" not in libs, (
            "Browser plugin's execute-time rule fires whenever Browser is "
            "imported; discovery must mirror"
        )
        assert "Browser" in libs


@pytest.mark.asyncio
class TestExplicitPreferenceUnchanged:
    """When explicit_library_preference IS set, OBS-20 changes nothing —
    existing session-preference path applies."""

    async def test_explicit_preference_overrides_imported_fallback(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = "SeleniumLibrary"  # explicit
        sess.imported_libraries = ["Browser", "SeleniumLibrary"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", session_id="s",
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # Browser excluded because explicit preference is SL
        assert "Browser" not in libs
        assert "SeleniumLibrary" in libs
        # Source is "session" (the explicit preference path), not
        # the imported-fallback path
        assert result["library_filter"]["source"] == "session"


@pytest.mark.asyncio
class TestNeutralLibrariesUnaffected:
    """BuiltIn / Collections / etc. always pass through, regardless
    of which filter rule fires."""

    async def test_imported_fallback_keeps_builtin(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        sess.imported_libraries = ["Browser", "BuiltIn"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Log", "library": "BuiltIn",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", session_id="s",
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # Compatible neutral library (BuiltIn) stays visible
        assert "BuiltIn" in libs
        assert "Browser" in libs
        assert "SeleniumLibrary" not in libs


@pytest.mark.asyncio
class TestNoImportFallback:
    """Sessions that import neither Browser nor SL → no filter."""

    async def test_no_browser_no_sl_no_filter(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        sess.imported_libraries = ["BuiltIn", "Collections"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", session_id="s",
        )
        # No filter applied
        assert "library_filter" not in result


@pytest.mark.asyncio
class TestLibraryNamePrecedence:
    """library_name parameter takes precedence over the imported-
    fallback (existing precedence preserved)."""

    async def test_library_name_overrides_imported_fallback(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        sess.imported_libraries = ["Browser", "BuiltIn"]
        patched_engines["sess_mgr"].get_session = MagicMock(return_value=sess)

        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        # library_name=SeleniumLibrary overrides session imported-fallback
        result = await _fn(find_keywords)(
            query="click", strategy="semantic",
            library_name="SeleniumLibrary", session_id="s",
        )
        # Browser excluded (library_name=SL filter); SL kept
        libs = {m["library"] for m in result["result"]["matches"]}
        assert "Browser" not in libs
        assert "SeleniumLibrary" in libs
        assert result["library_filter"]["source"] == "library_name"


@pytest.mark.asyncio
class TestNoSessionUnchanged:
    """When no session_id is provided, the imported-fallback path
    can't fire — behaviour is unchanged from pre-OBS-20."""

    async def test_no_session_id_no_imported_fallback(self, patched_engines):
        patched_engines["discover"].return_value = _stub_semantic([
            {"keyword_name": "Click", "library": "Browser",
             "confidence": 0.85, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary",
             "confidence": 0.80, "arguments": [], "argument_types": [],
             "documentation": "", "usage_example": ""},
        ])
        # No session_id, no library_name → no filter
        result = await _fn(find_keywords)(
            query="click", strategy="semantic",
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # Both libraries' matches remain
        assert "Browser" in libs
        assert "SeleniumLibrary" in libs
        assert "library_filter" not in result
