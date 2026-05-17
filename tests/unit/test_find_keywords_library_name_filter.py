"""find_keywords — ``library_name`` parameter must filter ALL strategies.

The 2026-05-17 defect report exposed a silent failure: an MCP call

    {
      "query": "select dropdown option by label or text",
      "strategy": "semantic",
      "context": "web",
      "library_name": "Browser",
      "limit": 10
    }

in a Browser-only session returned 7 SeleniumLibrary keywords among the
top 10 matches (with "Set Window Position" — completely unrelated — at
the top of the list with confidence 0.82). The ``library_name`` parameter
was accepted at the API surface, documented as "Optional library filter
for catalog search", and silently ignored for semantic/pattern. Agents
reading the response reasonably assume Browser library doesn't have a
``Select Dropdown`` keyword and write a SL-style call that breaks at
execute_step.

The fix wires ``library_name`` through to the existing
``_filter_keywords_by_session_library`` filter for all strategies, with
precedence ``library_name > session.explicit_library_preference > none``.
The filter uses the plugin-driven incompatibility table — for
``library_name="Browser"`` this excludes ``SeleniumLibrary`` and passes
through ``BuiltIn``, ``Collections``, ``String``, etc.

These tests pin:

(1) Reproducer: semantic + library_name="Browser", no session →
    SeleniumLibrary matches excluded; ``excluded_keywords`` populated;
    ``library_filter.source="library_name"``
(2) Symmetry: semantic + library_name="SeleniumLibrary" →
    Browser matches excluded
(3) Strategy parity: pattern + library_name="Browser" → same filter
(4) Precedence: library_name="Browser" + session preference
    "SeleniumLibrary" → library_name wins
(5) Fall-through: no library_name + session preference "Browser" →
    session wins; library_filter.source="session"
(6) Idempotency: neither set → unchanged behaviour, no library_filter
    field, no excluded_keywords
(7) Unknown library: library_name="FakeLib" → no exclusions
(8) Sibling libraries preserved: library_name="Browser" + a BuiltIn
    keyword in results → BuiltIn kept
(9) Docstring contract: parameter still documented and reachable
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _async_tool_fn(tool):
    """Unwrap a FastMCP tool to its underlying async function."""
    return getattr(tool, "fn", tool)


# ---------------------------------------------------------------------------
# Synthetic keyword fixtures — match the shape returned by the real engines
# ---------------------------------------------------------------------------


def _semantic_matches() -> List[Dict[str, Any]]:
    """Shape returned by ``KeywordMatcher.discover_keywords`` (under .matches)."""
    return [
        {
            "keyword_name": "Set Window Position",
            "library": "SeleniumLibrary",
            "confidence": 0.82,
            "arguments": ["x", "y"],
            "argument_types": ["int", "int"],
            "documentation": "Sets window position.",
            "usage_example": "Set Window Position    arg1    arg2",
        },
        {
            "keyword_name": "Click",
            "library": "Browser",
            "confidence": 0.80,
            "arguments": ["selector", "button"],
            "argument_types": ["str", "MouseButton"],
            "documentation": "Simulates click on element.",
            "usage_example": "Click    arg1    arg2",
        },
        {
            "keyword_name": "Click Element",
            "library": "SeleniumLibrary",
            "confidence": 0.80,
            "arguments": ["locator"],
            "argument_types": ["str"],
            "documentation": "Clicks element.",
            "usage_example": "Click Element    id=x",
        },
        {
            "keyword_name": "Select Options By",
            "library": "Browser",
            "confidence": 0.77,
            "arguments": ["selector", "attribute", "values"],
            "argument_types": ["str", "SelectAttribute", "str"],
            "documentation": "Select <option>.",
            "usage_example": "Select Options By    arg1    arg2    arg3",
        },
        {
            "keyword_name": "Select From List By Label",
            "library": "SeleniumLibrary",
            "confidence": 0.76,
            "arguments": ["locator", "labels"],
            "argument_types": ["str", "str"],
            "documentation": "Select by label.",
            "usage_example": "Select From List By Label    id=x    arg2",
        },
        {
            "keyword_name": "Log",
            "library": "BuiltIn",
            "confidence": 0.60,
            "arguments": ["message"],
            "argument_types": ["str"],
            "documentation": "Logs a message.",
            "usage_example": "Log    arg1",
        },
    ]


def _pattern_matches() -> List[Dict[str, Any]]:
    """Shape returned by ``execution_engine.search_keywords``."""
    return [
        {"name": "Click", "library": "Browser", "arguments": ["selector"]},
        {"name": "Click Element", "library": "SeleniumLibrary", "arguments": ["locator"]},
        {"name": "Click Button", "library": "SeleniumLibrary", "arguments": ["locator"]},
        {"name": "Tap", "library": "Browser", "arguments": ["selector"]},
        {"name": "Log", "library": "BuiltIn", "arguments": ["message"]},
    ]


# ---------------------------------------------------------------------------
# Test infrastructure: patch the matcher + engine + session lookup
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_engines():
    """Stub the semantic matcher, pattern engine, and ensure-all-libraries
    helper so tests don't hit the real RF runtime.

    Yields a context dict the tests can use to override per-case (e.g.
    to inject a session with a preference)."""
    semantic_payload = {
        "success": True,
        "action_description": "test action",
        "action_type": "click",
        "matches": _semantic_matches(),
        "total_matches": len(_semantic_matches()),
        "recommendations": [],
    }

    async def _ensure_loaded():
        return None

    discover_mock = AsyncMock(return_value=semantic_payload)
    search_mock = MagicMock(return_value=_pattern_matches())
    externalize_mock = MagicMock(side_effect=lambda _tool, _sid, r: r)
    track_mock = MagicMock()

    # session_manager.get_session returns None by default (no session).
    # Tests that need a session preference re-patch this.
    session_manager_mock = MagicMock()
    session_manager_mock.get_session = MagicMock(return_value=None)

    execution_engine_mock = MagicMock()
    execution_engine_mock.session_manager = session_manager_mock
    execution_engine_mock.search_keywords = search_mock

    with patch(
        "robotmcp.server.keyword_matcher.discover_keywords", discover_mock,
    ), patch(
        "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
    ), patch(
        "robotmcp.server._externalize_response", externalize_mock,
    ), patch(
        "robotmcp.server._track_tool_result", track_mock,
    ), patch(
        "robotmcp.server.execution_engine", execution_engine_mock,
    ):
        yield {
            "session_manager": session_manager_mock,
            "discover_mock": discover_mock,
            "search_mock": search_mock,
        }


# ---------------------------------------------------------------------------
# 1. Reproducer: the exact failing user payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReproducerSemanticBrowser:
    """The exact call from the 2026-05-17 defect report must now exclude
    SeleniumLibrary matches and surface ``library_filter``."""

    async def test_semantic_browser_filter_excludes_selenium(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="select dropdown option by label or text",
            strategy="semantic",
            context="web",
            library_name="Browser",
            limit=10,
        )
        matches = result["result"]["matches"]
        libs = sorted({m["library"] for m in matches})
        assert "SeleniumLibrary" not in libs, (
            f"Browser library_name must exclude SL; got libs={libs!r}"
        )
        # Sibling libraries (BuiltIn) are preserved.
        assert "Browser" in libs
        assert "BuiltIn" in libs

    async def test_excluded_keywords_populated(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="select dropdown option by label or text",
            strategy="semantic",
            context="web",
            library_name="Browser",
            limit=10,
        )
        excluded = result.get("excluded_keywords", [])
        assert excluded, "excluded_keywords must be populated"
        excluded_names = {e["keyword"] for e in excluded}
        assert {"Set Window Position", "Click Element", "Select From List By Label"}.issubset(
            excluded_names
        ), excluded_names

    async def test_library_filter_indicates_source(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["applied"] == "Browser"
        assert lf["source"] == "library_name"

    async def test_session_library_field_preserved_for_backcompat(self, patched_engines):
        """Legacy callers read ``session_library``. The new param must keep
        this populated whenever filtering applies."""
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        assert result.get("session_library") == "Browser"


# ---------------------------------------------------------------------------
# 2. Symmetry: SeleniumLibrary filter excludes Browser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSymmetrySeleniumFilter:
    """The filter must work symmetrically — library_name="SeleniumLibrary"
    excludes Browser, same plugin-table logic."""

    async def test_semantic_selenium_filter_excludes_browser(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="SeleniumLibrary",
        )
        matches = result["result"]["matches"]
        libs = sorted({m["library"] for m in matches})
        assert "Browser" not in libs, libs
        assert "SeleniumLibrary" in libs


# ---------------------------------------------------------------------------
# 3. Strategy parity: pattern branch honours library_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStrategyParityPattern:
    """Pattern strategy must apply the same filter."""

    async def test_pattern_browser_filter_excludes_selenium(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="click*",
            strategy="pattern",
            library_name="Browser",
        )
        matches = result.get("results", [])
        libs = sorted({m["library"] for m in matches})
        assert "SeleniumLibrary" not in libs, libs
        assert "Browser" in libs
        assert "BuiltIn" in libs
        # Response shape consistent with semantic branch.
        assert result.get("library_filter") == {
            "applied": "Browser",
            "source": "library_name",
        }


# ---------------------------------------------------------------------------
# 4. Precedence: library_name wins over session preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPrecedenceLibraryNameOverSession:
    """When BOTH library_name parameter AND session.explicit_library_preference
    are set, the parameter takes precedence (per-call override)."""

    async def test_library_name_overrides_session_preference(self, patched_engines):
        # Session preference set to SeleniumLibrary; per-call library_name
        # passed as Browser. Browser wins → SL excluded.
        sess = MagicMock()
        sess.explicit_library_preference = "SeleniumLibrary"
        patched_engines["session_manager"].get_session = MagicMock(return_value=sess)

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            session_id="sess-1",
            library_name="Browser",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" not in libs, libs
        assert "Browser" in libs
        # library_filter clearly attributes the source.
        assert result["library_filter"]["source"] == "library_name"
        assert result["library_filter"]["applied"] == "Browser"


# ---------------------------------------------------------------------------
# 5. Fall-through: no library_name → session preference applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFallthroughSessionPreference:
    """The existing session-preference path must still work when
    library_name is not supplied."""

    async def test_session_preference_applies_when_no_library_name(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = "Browser"
        patched_engines["session_manager"].get_session = MagicMock(return_value=sess)

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            session_id="sess-1",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" not in libs, libs
        assert "Browser" in libs
        # Source attributed to session, not library_name.
        assert result["library_filter"]["source"] == "session"
        assert result["library_filter"]["applied"] == "Browser"


# ---------------------------------------------------------------------------
# 6. Idempotency: no filter when neither is set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIdempotencyNoFilter:
    """When no library_name AND no session preference, behaviour is
    byte-for-byte the pre-fix shape: no library_filter, no
    excluded_keywords."""

    async def test_no_filter_when_neither_set(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
        )
        # Both library_name and SL keywords remain.
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" in libs
        assert "Browser" in libs
        assert "BuiltIn" in libs
        # No filter side-channels.
        assert "library_filter" not in result
        assert "excluded_keywords" not in result
        assert "session_library" not in result

    async def test_no_filter_when_session_has_no_preference(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        patched_engines["session_manager"].get_session = MagicMock(return_value=sess)
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            session_id="sess-1",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" in libs
        assert "library_filter" not in result


# ---------------------------------------------------------------------------
# 7. Unknown library: no exclusions (graceful)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnknownLibrary:
    """An unknown library_name produces no exclusions (the plugin table
    returns an empty incompatibility list). The filter is a no-op rather
    than an error — agents are not blocked by a typo."""

    async def test_unknown_library_no_exclusions(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="NoSuchLibrary",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        # All original libraries remain — no incompatibility table for
        # the unknown name.
        assert "SeleniumLibrary" in libs
        assert "Browser" in libs
        # No exclusions surface, so the response carries no library_filter
        # diagnostic (parallel to the no-filter idempotency case).
        assert "excluded_keywords" not in result
        assert "library_filter" not in result


# ---------------------------------------------------------------------------
# 8. Sibling libraries preserved (BuiltIn, Collections, String)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSiblingLibrariesPreserved:
    """library_name="Browser" must NOT exclude BuiltIn/Collections/etc. —
    only the explicitly-incompatible siblings (SeleniumLibrary)."""

    async def test_builtin_preserved_under_browser_filter(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        names = {m["keyword_name"] for m in result["result"]["matches"]}
        # ``Log`` is BuiltIn — must remain after filtering.
        assert "Log" in names, (
            f"BuiltIn keywords must not be excluded by Browser filter; "
            f"got names={sorted(names)!r}"
        )


# ---------------------------------------------------------------------------
# 9. Docstring contract — the parameter is reachable and named
# ---------------------------------------------------------------------------


class TestDocstringContract:
    """Light contract check: the parameter exists in the signature and
    the docstring no longer says "catalog search" (the old narrow
    promise)."""

    def test_library_name_parameter_present(self):
        import inspect
        sig = inspect.signature(_async_tool_fn(find_keywords))
        assert "library_name" in sig.parameters
        assert sig.parameters["library_name"].default is None

    def test_docstring_announces_all_strategies(self):
        fn = _async_tool_fn(find_keywords)
        doc = (fn.__doc__ or "").lower()
        # Old narrow docstring said "for catalog search". The new doc
        # explicitly says "ALL strategies".
        assert "all strategies" in doc, (
            f"docstring should announce all-strategies coverage; got: {doc[:400]}"
        )
