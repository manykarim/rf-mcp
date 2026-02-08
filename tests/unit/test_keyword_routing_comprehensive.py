"""Comprehensive keyword routing unit tests.

Covers: session-aware routing, ambiguous keyword detection, plugin validation,
library preference inference, keyword overrides, and error cases.

Run with: uv run pytest tests/unit/test_keyword_routing_comprehensive.py -v
"""

__test__ = True

from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.models.library_models import KeywordInfo, LibraryInfo
from robotmcp.models.session_models import ExecutionSession
from robotmcp.plugins.manager import LibraryPluginManager
from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin


def _make_keyword_info(name: str, library: str) -> KeywordInfo:
    """Create a KeywordInfo with minimal fields."""
    return KeywordInfo(
        name=name,
        library=library,
        method_name=name.lower().replace(" ", "_"),
    )


def _make_lib_info(name: str, keywords: Dict[str, str] = None) -> LibraryInfo:
    """Create a LibraryInfo with optional keyword map.

    Args:
        name: Library name
        keywords: dict of {keyword_name: library_name} for KeywordInfo creation
    """
    lib = LibraryInfo(name=name, instance=None)
    if keywords:
        for kw_name, kw_lib in keywords.items():
            lib.keywords[kw_name] = _make_keyword_info(kw_name, kw_lib or name)
    return lib


def _make_session(
    session_id: str = "test",
    imported_libraries: list = None,
    loaded_libraries: set = None,
    search_order: list = None,
    explicit_library_preference: str = None,
) -> ExecutionSession:
    """Create a real ExecutionSession with specified configuration."""
    session = ExecutionSession(session_id=session_id)
    if imported_libraries is not None:
        session.imported_libraries = imported_libraries
    if loaded_libraries is not None:
        session.loaded_libraries = loaded_libraries
    if search_order is not None:
        session.search_order = search_order
    if explicit_library_preference is not None:
        session.explicit_library_preference = explicit_library_preference
    return session


# =============================================================================
# GAP-1: KeywordDiscovery.find_keyword() with session_libraries
# =============================================================================


class TestSessionAwareKeywordSearch:
    """Test find_keyword with session_libraries filtering."""

    def test_find_keyword_filters_by_session_libraries(self):
        """Only keywords from session libraries should be found."""
        kd = KeywordDiscovery()

        # Add keywords from multiple libraries
        builtin = _make_lib_info("BuiltIn", {"Log": "BuiltIn", "Should Be Equal": "BuiltIn"})
        browser = _make_lib_info("Browser", {"Click": "Browser", "Fill Text": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Click Element": "SeleniumLibrary"})

        kd.add_keywords_to_cache(builtin)
        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        # With session_libraries=["BuiltIn", "Browser"], Click Element should NOT be found
        result = kd.find_keyword("Click Element", session_libraries=["BuiltIn", "Browser"])
        # Should either not find it or find something else
        if result is not None:
            assert result.library != "SeleniumLibrary"

    def test_find_keyword_prefers_session_library_by_priority(self):
        """When keyword exists in multiple session libraries, prefer by load_priority."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"Go To": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Go To": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        # Browser has priority 5, SeleniumLibrary has priority 8
        # So Browser should win when both are session libraries
        result = kd.find_keyword("Go To", session_libraries=["Browser", "SeleniumLibrary"])
        assert result is not None
        assert result.library == "Browser"

    def test_find_keyword_no_session_libs_uses_full_cache(self):
        """Without session_libraries, search the full cache."""
        kd = KeywordDiscovery()
        selenium = _make_lib_info("SeleniumLibrary", {"Click Element": "SeleniumLibrary"})
        kd.add_keywords_to_cache(selenium)

        result = kd.find_keyword("Click Element")
        assert result is not None
        assert result.name == "Click Element"

    def test_find_keyword_active_library_takes_priority(self):
        """active_library parameter uses library-specific key first."""
        kd = KeywordDiscovery()
        browser = _make_lib_info("Browser", {"Click": "Browser"})
        kd.add_keywords_to_cache(browser)

        result = kd.find_keyword("Click", active_library="Browser")
        assert result is not None
        assert result.library == "Browser"

    def test_find_keyword_empty_string_returns_none(self):
        """Empty keyword name returns None."""
        kd = KeywordDiscovery()
        assert kd.find_keyword("") is None
        assert kd.find_keyword(None) is None

    def test_find_keyword_session_libs_excludes_unmatched_library(self):
        """Keywords from libraries not in session_libraries are excluded."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"New Page": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Open Browser": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        # Only SeleniumLibrary in session
        result = kd.find_keyword("New Page", session_libraries=["SeleniumLibrary"])
        # New Page is only in Browser, so should not be found via session filtering
        if result is not None:
            assert result.library != "Browser"

    def test_find_keyword_active_library_with_ambiguous(self):
        """active_library resolves ambiguity between two libraries."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"Go To": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Go To": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        result = kd.find_keyword("Go To", active_library="SeleniumLibrary")
        assert result is not None
        assert result.library == "SeleniumLibrary"


# =============================================================================
# GAP-2: Ambiguous/shadowed keyword detection
# =============================================================================


class TestAmbiguousKeywordDetection:
    """Test ambiguous keyword tracking and detection."""

    def test_same_keyword_from_two_libraries_is_ambiguous(self):
        """Adding same keyword from 2 libraries marks it ambiguous."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"Go To": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Go To": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        assert kd.is_keyword_ambiguous("go to") is True
        assert "go to" in kd.ambiguous_keywords

    def test_unique_keyword_not_ambiguous(self):
        """Keyword unique to one library is not ambiguous."""
        kd = KeywordDiscovery()
        browser = _make_lib_info("Browser", {"New Page": "Browser"})
        kd.add_keywords_to_cache(browser)

        assert kd.is_keyword_ambiguous("new page") is False

    def test_shadowed_keywords_tracked(self):
        """Shadowed keywords contain both library entries."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"Close Browser": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Close Browser": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        shadow_info = kd.get_keyword_shadow_info()
        assert shadow_info["ambiguous_count"] >= 1
        assert "close browser" in shadow_info["ambiguous_keywords"]

    def test_shadow_info_contains_library_sources(self):
        """Shadow info should list which libraries provide the keyword."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"Press Keys": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Press Keys": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        shadow_info = kd.get_keyword_shadow_info()
        shadows = shadow_info.get("shadows", {})
        assert "press keys" in shadows
        libs = [lib for lib, _ in shadows["press keys"]]
        assert "Browser" in libs
        assert "SeleniumLibrary" in libs

    def test_is_keyword_ambiguous_case_insensitive(self):
        """Ambiguity check should be case-insensitive."""
        kd = KeywordDiscovery()

        browser = _make_lib_info("Browser", {"Get Title": "Browser"})
        selenium = _make_lib_info("SeleniumLibrary", {"Get Title": "SeleniumLibrary"})

        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(selenium)

        assert kd.is_keyword_ambiguous("get title") is True
        assert kd.is_keyword_ambiguous("GET TITLE") is True

    def test_three_libraries_same_keyword_all_tracked(self):
        """Three libraries with same keyword should all appear in shadows."""
        kd = KeywordDiscovery()

        lib_a = _make_lib_info("LibA", {"Log": "LibA"})
        lib_b = _make_lib_info("LibB", {"Log": "LibB"})
        lib_c = _make_lib_info("LibC", {"Log": "LibC"})

        kd.add_keywords_to_cache(lib_a)
        kd.add_keywords_to_cache(lib_b)
        kd.add_keywords_to_cache(lib_c)

        assert kd.is_keyword_ambiguous("log") is True
        shadow_info = kd.get_keyword_shadow_info()
        shadows = shadow_info["shadows"]
        assert "log" in shadows
        libs = [lib for lib, _ in shadows["log"]]
        # At minimum LibA and LibB must be tracked (LibA is the first shadow entry)
        assert len(libs) >= 2


# =============================================================================
# GAP-4: LibraryPluginManager.get_library_for_keyword()
# =============================================================================


class TestPluginManagerKeywordMap:
    """Test LibraryPluginManager keyword map."""

    def test_browser_plugin_keyword_map(self):
        """BrowserLibraryPlugin maps known keywords to Browser."""
        manager = LibraryPluginManager()
        plugin = BrowserLibraryPlugin()
        manager.register_plugin(plugin)

        assert manager.get_library_for_keyword("new browser") == "Browser"
        assert manager.get_library_for_keyword("new page") == "Browser"
        assert manager.get_library_for_keyword("get page source") == "Browser"

    def test_selenium_plugin_keyword_map(self):
        """SeleniumLibraryPlugin maps known keywords to SeleniumLibrary."""
        manager = LibraryPluginManager()
        plugin = SeleniumLibraryPlugin()
        manager.register_plugin(plugin)

        assert manager.get_library_for_keyword("get source") == "SeleniumLibrary"

    def test_unknown_keyword_returns_none(self):
        """Unknown keyword returns None from plugin manager."""
        manager = LibraryPluginManager()
        plugin = BrowserLibraryPlugin()
        manager.register_plugin(plugin)

        assert manager.get_library_for_keyword("totally unknown keyword") is None

    def test_keyword_map_case_insensitive(self):
        """Keyword map lookup should be case-insensitive."""
        manager = LibraryPluginManager()
        plugin = BrowserLibraryPlugin()
        manager.register_plugin(plugin)

        assert manager.get_library_for_keyword("New Browser") == "Browser"
        assert manager.get_library_for_keyword("NEW BROWSER") == "Browser"

    def test_both_plugins_registered(self):
        """Both plugins can be registered without conflict."""
        manager = LibraryPluginManager()
        manager.register_plugin(BrowserLibraryPlugin())
        manager.register_plugin(SeleniumLibraryPlugin())

        assert manager.get_library_for_keyword("new browser") == "Browser"
        assert manager.get_library_for_keyword("get source") == "SeleniumLibrary"

    def test_plugin_manager_lists_both_plugins(self):
        """Both plugins appear in list_plugin_names."""
        manager = LibraryPluginManager()
        manager.register_plugin(BrowserLibraryPlugin())
        manager.register_plugin(SeleniumLibraryPlugin())

        names = manager.list_plugin_names()
        assert "Browser" in names
        assert "SeleniumLibrary" in names

    def test_qualified_browser_keyword_map(self):
        """Qualified 'browser.new browser' key also maps to Browser."""
        manager = LibraryPluginManager()
        manager.register_plugin(BrowserLibraryPlugin())

        # The plugin registers both qualified and unqualified forms
        assert manager.get_library_for_keyword("browser.new browser") == "Browser"
        assert manager.get_library_for_keyword("browser.new page") == "Browser"

    def test_qualified_selenium_keyword_map(self):
        """Qualified 'seleniumlibrary.get source' key also maps to SeleniumLibrary."""
        manager = LibraryPluginManager()
        manager.register_plugin(SeleniumLibraryPlugin())

        assert manager.get_library_for_keyword("seleniumlibrary.get source") == "SeleniumLibrary"


# =============================================================================
# GAP-5/6: Browser/Selenium plugin validation asymmetry
# =============================================================================


class TestBrowserPluginValidation:
    """Test BrowserLibraryPlugin.validate_keyword_for_session."""

    def test_browser_plugin_blocks_selenium_keyword(self):
        """Browser plugin blocks SeleniumLibrary keyword with alternative."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference="Browser",
            imported_libraries=["Browser", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Input Text", "SeleniumLibrary")
        assert result is not None
        assert result["success"] is False
        assert "alternative" in result

    def test_browser_plugin_allows_shared_keyword(self):
        """Browser plugin allows shared keyword (Go To) from SeleniumLibrary."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference="Browser",
            imported_libraries=["Browser", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Go To", "SeleniumLibrary")
        assert result is None  # None means "allowed"

    def test_browser_plugin_triggers_on_imported_libraries(self):
        """Browser plugin validates even without explicit_library_preference (via imported_libraries)."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference=None,  # No explicit preference
            imported_libraries=["Browser", "BuiltIn"],  # But Browser is imported
        )

        result = plugin.validate_keyword_for_session(session, "Input Text", "SeleniumLibrary")
        # Should block because Browser is in imported_libraries
        assert result is not None
        assert result["success"] is False

    def test_browser_plugin_skips_when_not_imported(self):
        """Browser plugin doesn't validate when Browser not in session."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference=None,
            imported_libraries=["BuiltIn"],  # No Browser
        )

        result = plugin.validate_keyword_for_session(session, "Input Text", "SeleniumLibrary")
        assert result is None  # Should not block

    def test_browser_plugin_skips_when_preference_is_not_browser(self):
        """Browser plugin returns None when explicit preference is non-Browser."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference="SeleniumLibrary",
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Input Text", "SeleniumLibrary")
        assert result is None

    def test_browser_plugin_allows_non_selenium_source(self):
        """Browser plugin allows keywords from non-SeleniumLibrary sources."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference="Browser",
            imported_libraries=["Browser", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Log", "BuiltIn")
        assert result is None

    def test_browser_plugin_blocks_with_known_alternative(self):
        """Browser plugin returns alternative info for known keyword alternatives."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference="Browser",
            imported_libraries=["Browser", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "click element", "SeleniumLibrary")
        assert result is not None
        assert result["success"] is False
        assert result["alternative"] == "Click"

    def test_browser_plugin_blocks_without_known_alternative(self):
        """Browser plugin blocks unknown SeleniumLibrary keyword gracefully."""
        plugin = BrowserLibraryPlugin()
        session = _make_session(
            explicit_library_preference="Browser",
            imported_libraries=["Browser", "BuiltIn"],
        )

        # "Execute Javascript" is a SeleniumLibrary keyword not in KEYWORD_ALTERNATIVES
        result = plugin.validate_keyword_for_session(
            session, "Execute Javascript", "SeleniumLibrary"
        )
        assert result is not None
        assert result["success"] is False
        # alternative should be None since it is not in the alternatives dict
        assert result["alternative"] is None


class TestSeleniumPluginValidation:
    """Test SeleniumLibraryPlugin.validate_keyword_for_session."""

    def test_selenium_plugin_blocks_browser_keyword(self):
        """Selenium plugin blocks Browser keyword with alternative."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference="SeleniumLibrary",
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "New Browser", "Browser")
        assert result is not None
        assert result["success"] is False
        assert "alternative" in result

    def test_selenium_plugin_allows_shared_keyword(self):
        """Selenium plugin allows shared keyword from Browser."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference="SeleniumLibrary",
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Go To", "Browser")
        assert result is None  # Allowed

    def test_selenium_plugin_requires_explicit_preference(self):
        """Selenium plugin ONLY triggers on explicit_library_preference (not imported_libraries)."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference=None,  # No explicit preference
            imported_libraries=["SeleniumLibrary", "BuiltIn"],  # Selenium is imported
        )

        result = plugin.validate_keyword_for_session(session, "New Browser", "Browser")
        # Selenium plugin does NOT block because no explicit preference
        assert result is None

    def test_selenium_plugin_skips_non_selenium_preference(self):
        """Selenium plugin doesn't validate when preference is Browser."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference="Browser",
            imported_libraries=["Browser", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "New Browser", "Browser")
        assert result is None

    def test_selenium_plugin_allows_non_browser_source(self):
        """Selenium plugin allows keywords from non-Browser sources."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference="SeleniumLibrary",
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Log", "BuiltIn")
        assert result is None

    def test_selenium_plugin_blocks_known_browser_alternative(self):
        """Selenium plugin returns alternative for known Browser keyword."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference="SeleniumLibrary",
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(session, "Fill Text", "Browser")
        assert result is not None
        assert result["success"] is False
        assert result["alternative"] == "Input Text"

    def test_selenium_plugin_blocks_unknown_browser_keyword(self):
        """Selenium plugin blocks unknown Browser keyword gracefully."""
        plugin = SeleniumLibraryPlugin()
        session = _make_session(
            explicit_library_preference="SeleniumLibrary",
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
        )

        result = plugin.validate_keyword_for_session(
            session, "Some Unknown Keyword", "Browser"
        )
        assert result is not None
        assert result["success"] is False
        assert result["alternative"] is None


# =============================================================================
# GAP-9: ExecutionSession.resolve_keyword_library()
# =============================================================================


class TestResolveKeywordLibrary:
    """Test ExecutionSession.resolve_keyword_library."""

    def test_resolve_returns_first_loaded_in_search_order(self):
        """resolve_keyword_library returns first loaded library in search order."""
        session = _make_session(
            search_order=["Browser", "BuiltIn"],
            loaded_libraries={"Browser", "BuiltIn"},
        )
        result = session.resolve_keyword_library("Click")
        assert result == "Browser"

    def test_resolve_skips_unloaded_libraries(self):
        """Libraries in search_order but not in loaded_libraries are skipped."""
        session = _make_session(
            search_order=["Browser", "BuiltIn"],
            loaded_libraries={"BuiltIn"},  # Browser not loaded
        )
        result = session.resolve_keyword_library("Click")
        assert result == "BuiltIn"

    def test_resolve_falls_back_to_builtin(self):
        """Empty search order falls back to BuiltIn."""
        session = _make_session(
            search_order=[],
            loaded_libraries=set(),
        )
        result = session.resolve_keyword_library("anything")
        assert result == "BuiltIn"

    def test_resolve_strips_library_prefix(self):
        """Library.Keyword format has prefix stripped."""
        session = _make_session(
            search_order=["Browser", "BuiltIn"],
            loaded_libraries={"Browser", "BuiltIn"},
        )
        result = session.resolve_keyword_library("Browser.Click")
        assert result == "Browser"

    def test_resolve_with_selenium_in_search_order(self):
        """SeleniumLibrary in search_order and loaded returns SeleniumLibrary."""
        session = _make_session(
            search_order=["SeleniumLibrary", "BuiltIn"],
            loaded_libraries={"SeleniumLibrary", "BuiltIn"},
        )
        result = session.resolve_keyword_library("Click Element")
        assert result == "SeleniumLibrary"

    def test_resolve_search_order_determines_priority(self):
        """First loaded library in search_order wins regardless of name."""
        session = _make_session(
            search_order=["BuiltIn", "Browser"],
            loaded_libraries={"Browser", "BuiltIn"},
        )
        # BuiltIn is first in search_order and loaded
        result = session.resolve_keyword_library("Click")
        assert result == "BuiltIn"


# =============================================================================
# GAP-12: Error cases
# =============================================================================


class TestKeywordRoutingErrorCases:
    """Test error handling in keyword routing."""

    def test_plugin_manager_validate_with_none_session_library(self):
        """validate_keyword_for_session with None session_library returns None."""
        manager = LibraryPluginManager()
        manager.register_plugin(BrowserLibraryPlugin())

        result = manager.validate_keyword_for_session(None, MagicMock(), "Click", "Browser")
        assert result is None

    def test_plugin_manager_override_with_none_library(self):
        """get_keyword_override with None library returns None."""
        manager = LibraryPluginManager()
        result = manager.get_keyword_override(None, "Click")
        assert result is None

    def test_plugin_manager_hints_with_unregistered_library(self):
        """generate_failure_hints with unregistered library returns empty."""
        manager = LibraryPluginManager()
        result = manager.generate_failure_hints("Unknown", MagicMock(), "kw", [], "error")
        assert result == []

    def test_plugin_manager_hints_with_none_library(self):
        """generate_failure_hints with None library returns empty."""
        manager = LibraryPluginManager()
        result = manager.generate_failure_hints(None, MagicMock(), "kw", [], "error")
        assert result == []

    def test_keyword_discovery_empty_cache_returns_none(self):
        """find_keyword on empty cache returns None."""
        kd = KeywordDiscovery()
        assert kd.find_keyword("anything") is None

    def test_keyword_discovery_get_keywords_by_unknown_library(self):
        """get_keywords_by_library for unknown library returns empty."""
        kd = KeywordDiscovery()
        result = kd.get_keywords_by_library("UnknownLib")
        assert result == []

    def test_plugin_manager_override_with_unregistered_library(self):
        """get_keyword_override for unregistered library returns None."""
        manager = LibraryPluginManager()
        result = manager.get_keyword_override("NonExistent", "open browser")
        assert result is None

    def test_plugin_manager_run_before_keyword_with_none_library(self):
        """run_before_keyword_execution with None library does nothing."""
        manager = LibraryPluginManager()
        # Should not raise
        manager.run_before_keyword_execution(
            None, MagicMock(), "Click", MagicMock(), MagicMock()
        )

    def test_plugin_manager_run_before_keyword_with_unregistered_library(self):
        """run_before_keyword_execution with unregistered library does nothing."""
        manager = LibraryPluginManager()
        # Should not raise
        manager.run_before_keyword_execution(
            "Unknown", MagicMock(), "Click", MagicMock(), MagicMock()
        )


# =============================================================================
# GAP-15: Browser plugin edge cases with both libraries
# =============================================================================


class TestBrowserPluginBothImportedEdgeCases:
    """Test Browser plugin validation when both libraries imported."""

    def test_open_browser_override_registered(self):
        """Open Browser override should be registered in the plugin."""
        plugin = BrowserLibraryPlugin()
        overrides = plugin.get_keyword_overrides()
        assert "open browser" in overrides

    def test_incompatible_libraries_returns_selenium(self):
        """Browser plugin reports SeleniumLibrary as incompatible."""
        plugin = BrowserLibraryPlugin()
        incompat = plugin.get_incompatible_libraries()
        assert "SeleniumLibrary" in incompat

    def test_selenium_incompatible_with_browser(self):
        """Selenium plugin reports Browser as incompatible."""
        plugin = SeleniumLibraryPlugin()
        incompat = plugin.get_incompatible_libraries()
        assert "Browser" in incompat

    def test_browser_plugin_shared_keywords_match_selenium(self):
        """Both plugins share the same set of shared keywords."""
        browser_shared = BrowserLibraryPlugin._SHARED_KEYWORDS
        selenium_shared = SeleniumLibraryPlugin._SHARED_KEYWORDS
        assert browser_shared == selenium_shared

    def test_browser_plugin_shared_keywords_contains_go_to(self):
        """Shared keywords should contain common keywords like 'go to'."""
        shared = BrowserLibraryPlugin._SHARED_KEYWORDS
        assert "go to" in shared
        assert "close browser" in shared
        assert "add cookie" in shared

    def test_browser_plugin_keyword_alternatives_has_entries(self):
        """Browser plugin has keyword alternatives defined."""
        plugin = BrowserLibraryPlugin()
        alts = plugin.get_keyword_alternatives()
        assert len(alts) > 0
        assert "input text" in alts
        assert "click element" in alts

    def test_selenium_plugin_keyword_alternatives_has_entries(self):
        """Selenium plugin has keyword alternatives defined."""
        plugin = SeleniumLibraryPlugin()
        alts = plugin.get_keyword_alternatives()
        assert len(alts) > 0
        assert "new browser" in alts
        assert "fill text" in alts

    def test_plugin_manager_incompatible_libraries_via_manager(self):
        """Plugin manager correctly delegates get_incompatible_libraries."""
        manager = LibraryPluginManager()
        manager.register_plugin(BrowserLibraryPlugin())
        manager.register_plugin(SeleniumLibraryPlugin())

        assert "SeleniumLibrary" in manager.get_incompatible_libraries("Browser")
        assert "Browser" in manager.get_incompatible_libraries("SeleniumLibrary")

    def test_plugin_manager_incompatible_unknown_returns_empty(self):
        """get_incompatible_libraries for unknown library returns empty."""
        manager = LibraryPluginManager()
        assert manager.get_incompatible_libraries("UnknownLib") == []


# =============================================================================
# Keyword Discovery helpers
# =============================================================================


class TestKeywordDiscoveryHelpers:
    """Test helper methods on KeywordDiscovery."""

    def test_method_to_keyword_name(self):
        """method_to_keyword_name converts snake_case to Title Case."""
        kd = KeywordDiscovery()
        assert kd.method_to_keyword_name("click_element") == "Click Element"
        assert kd.method_to_keyword_name("go_to") == "Go To"
        assert kd.method_to_keyword_name("log") == "Log"

    def test_is_dom_changing_keyword(self):
        """DOM-changing keywords are detected."""
        kd = KeywordDiscovery()
        assert kd.is_dom_changing_keyword("Click Element") is True
        assert kd.is_dom_changing_keyword("Go To") is True
        assert kd.is_dom_changing_keyword("Log") is False
        assert kd.is_dom_changing_keyword("Should Be Equal") is False

    def test_is_dom_changing_keyword_various_patterns(self):
        """All dom_changing_patterns are recognized."""
        kd = KeywordDiscovery()
        # From the dom_changing_patterns list
        assert kd.is_dom_changing_keyword("Fill Text") is True
        assert kd.is_dom_changing_keyword("Select Options") is True
        assert kd.is_dom_changing_keyword("Check Checkbox") is True
        assert kd.is_dom_changing_keyword("Submit Form") is True
        assert kd.is_dom_changing_keyword("Clear Element") is True
        assert kd.is_dom_changing_keyword("Upload File") is True
        assert kd.is_dom_changing_keyword("New Page") is True
        assert kd.is_dom_changing_keyword("Close Page") is True
        assert kd.is_dom_changing_keyword("Open Browser") is True
        assert kd.is_dom_changing_keyword("Reload Page") is True
        assert kd.is_dom_changing_keyword("Navigate Back") is True

    def test_get_keyword_count_reflects_cache(self):
        """get_keyword_count matches cache size."""
        kd = KeywordDiscovery()
        assert kd.get_keyword_count() == 0

        lib = _make_lib_info("TestLib", {"Keyword A": "TestLib", "Keyword B": "TestLib"})
        kd.add_keywords_to_cache(lib)
        # Each keyword gets both qualified and simple keys
        assert kd.get_keyword_count() >= 2

    def test_remove_keywords_from_cache(self):
        """remove_keywords_from_cache clears library-specific entries."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("TestLib", {"Keyword A": "TestLib"})
        kd.add_keywords_to_cache(lib)

        count_before = kd.get_keyword_count()
        removed = kd.remove_keywords_from_cache(lib)
        assert removed > 0
        assert kd.get_keyword_count() < count_before

    def test_get_keyword_suggestions(self):
        """get_keyword_suggestions returns partial matches."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("BuiltIn", {
            "Log": "BuiltIn",
            "Log Many": "BuiltIn",
            "Should Be Equal": "BuiltIn",
        })
        kd.add_keywords_to_cache(lib)

        suggestions = kd.get_keyword_suggestions("log")
        assert len(suggestions) >= 1

    def test_get_keyword_suggestions_empty_input(self):
        """get_keyword_suggestions with empty input returns empty."""
        kd = KeywordDiscovery()
        assert kd.get_keyword_suggestions("") == []

    def test_get_keyword_suggestions_respects_limit(self):
        """get_keyword_suggestions respects the limit parameter."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("TestLib", {
            f"Keyword {i}": "TestLib" for i in range(20)
        })
        kd.add_keywords_to_cache(lib)

        suggestions = kd.get_keyword_suggestions("keyword", limit=3)
        assert len(suggestions) <= 3

    def test_get_keywords_by_library(self):
        """get_keywords_by_library returns only keywords from that library."""
        kd = KeywordDiscovery()
        browser = _make_lib_info("Browser", {"Click": "Browser", "Fill Text": "Browser"})
        builtin = _make_lib_info("BuiltIn", {"Log": "BuiltIn"})
        kd.add_keywords_to_cache(browser)
        kd.add_keywords_to_cache(builtin)

        browser_kws = kd.get_keywords_by_library("Browser")
        assert len(browser_kws) == 2
        assert all(kw.library == "Browser" for kw in browser_kws)

        builtin_kws = kd.get_keywords_by_library("BuiltIn")
        assert len(builtin_kws) == 1
        assert builtin_kws[0].library == "BuiltIn"

    def test_get_all_keywords(self):
        """get_all_keywords returns all cached keyword info objects."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("TestLib", {"KW A": "TestLib", "KW B": "TestLib"})
        kd.add_keywords_to_cache(lib)

        all_kws = kd.get_all_keywords()
        assert len(all_kws) >= 2

    def test_find_keyword_fuzzy_match_underscore(self):
        """find_keyword matches keyword with underscores instead of spaces."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("BuiltIn", {"Should Be Equal": "BuiltIn"})
        kd.add_keywords_to_cache(lib)

        result = kd.find_keyword("should_be_equal")
        assert result is not None
        assert result.name == "Should Be Equal"

    def test_find_keyword_no_spaces_not_matched(self):
        """find_keyword does not match keyword with all spaces removed (no variation handles it)."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("BuiltIn", {"Should Be Equal": "BuiltIn"})
        kd.add_keywords_to_cache(lib)

        # "shouldbeequal" has no spaces/underscores/hyphens to convert,
        # and substring matching fails, so no match is returned.
        result = kd.find_keyword("shouldbeequal")
        assert result is None

    def test_find_keyword_fuzzy_match_hyphen(self):
        """find_keyword matches keyword with hyphens instead of spaces."""
        kd = KeywordDiscovery()
        lib = _make_lib_info("BuiltIn", {"Should Be Equal": "BuiltIn"})
        kd.add_keywords_to_cache(lib)

        result = kd.find_keyword("should-be-equal")
        assert result is not None
        assert result.name == "Should Be Equal"

    def test_create_short_doc_first_line_ending_with_period(self):
        """create_short_doc returns first line if it ends with period."""
        kd = KeywordDiscovery()
        # The method splits on literal '\\n' (escaped newline in source)
        doc = "First line.\\nSecond line."
        result = kd.create_short_doc(doc)
        assert result == "First line."

    def test_create_short_doc_single_line_with_period(self):
        """create_short_doc returns entire string if single line ends with period."""
        kd = KeywordDiscovery()
        assert kd.create_short_doc("Complete sentence.") == "Complete sentence."

    def test_create_short_doc_no_period_adds_one(self):
        """create_short_doc adds period if first line does not end with one."""
        kd = KeywordDiscovery()
        assert kd.create_short_doc("No period here") == "No period here."

    def test_create_short_doc_empty(self):
        """create_short_doc handles empty string."""
        kd = KeywordDiscovery()
        assert kd.create_short_doc("") == ""
