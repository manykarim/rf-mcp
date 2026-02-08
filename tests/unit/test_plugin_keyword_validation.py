"""Tests for plugin keyword validation and library compatibility."""

from __future__ import annotations

import pytest

from robotmcp.config import library_registry
from robotmcp.models.session_models import ExecutionSession
from robotmcp.plugins import get_library_plugin_manager
from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin


@pytest.fixture(autouse=True)
def reset_plugin_state():
    """Reset plugin manager and registry state before and after each test."""
    library_registry._reset_plugin_state_for_tests()  # type: ignore[attr-defined]
    yield
    library_registry._reset_plugin_state_for_tests()  # type: ignore[attr-defined]


class TestBrowserPluginValidation:
    """Test BrowserLibraryPlugin keyword validation."""

    def test_get_incompatible_libraries(self):
        """Browser Library is incompatible with SeleniumLibrary."""
        plugin = BrowserLibraryPlugin()
        incompatible = plugin.get_incompatible_libraries()

        assert "SeleniumLibrary" in incompatible
        assert len(incompatible) == 1

    def test_get_keyword_alternatives_contains_common_selenium_keywords(self):
        """Browser plugin provides alternatives for common SeleniumLibrary keywords."""
        plugin = BrowserLibraryPlugin()
        alternatives = plugin.get_keyword_alternatives()

        # Check that common SeleniumLibrary keywords have alternatives
        assert "open browser" in alternatives
        assert "input text" in alternatives
        assert "click element" in alternatives
        assert "click button" in alternatives

        # Check alternative structure
        open_browser = alternatives["open browser"]
        assert "alternative" in open_browser
        assert "example" in open_browser
        assert "explanation" in open_browser
        assert open_browser["alternative"] == "New Browser + New Page"

    def test_validate_keyword_for_browser_session_rejects_selenium_keyword(self):
        """Browser session should reject SeleniumLibrary-only keywords."""
        plugin = BrowserLibraryPlugin()

        # Create a session with Browser preference
        session = ExecutionSession(session_id="browser-test")
        session.explicit_library_preference = "Browser"

        # Validate a SeleniumLibrary-only keyword (not shared)
        result = plugin.validate_keyword_for_session(
            session, "Input Text", "SeleniumLibrary"
        )

        assert result is not None
        assert result["success"] is False
        assert "SeleniumLibrary" in result["error"]
        assert "Browser Library" in result["error"]
        assert result["alternative"] == "Fill Text"

    def test_validate_keyword_for_browser_session_accepts_browser_keyword(self):
        """Browser session should accept Browser Library keywords."""
        plugin = BrowserLibraryPlugin()

        session = ExecutionSession(session_id="browser-test")
        session.explicit_library_preference = "Browser"

        # Validate a Browser Library keyword - should return None (valid)
        result = plugin.validate_keyword_for_session(
            session, "New Browser", "Browser"
        )

        assert result is None  # None means valid

    def test_validate_keyword_for_non_browser_session_allows_any(self):
        """Non-Browser session should not validate keywords."""
        plugin = BrowserLibraryPlugin()

        # Create a session without Browser preference
        session = ExecutionSession(session_id="other-test")
        session.explicit_library_preference = None

        # Should return None (no validation needed)
        result = plugin.validate_keyword_for_session(
            session, "Open Browser", "SeleniumLibrary"
        )

        assert result is None


class TestSeleniumPluginValidation:
    """Test SeleniumLibraryPlugin keyword validation."""

    def test_get_incompatible_libraries(self):
        """SeleniumLibrary is incompatible with Browser Library."""
        plugin = SeleniumLibraryPlugin()
        incompatible = plugin.get_incompatible_libraries()

        assert "Browser" in incompatible
        assert len(incompatible) == 1

    def test_get_keyword_alternatives_contains_common_browser_keywords(self):
        """Selenium plugin provides alternatives for common Browser Library keywords."""
        plugin = SeleniumLibraryPlugin()
        alternatives = plugin.get_keyword_alternatives()

        # Check that common Browser Library keywords have alternatives
        assert "new browser" in alternatives
        assert "new page" in alternatives
        assert "fill text" in alternatives
        assert "click" in alternatives

        # Check alternative structure
        new_browser = alternatives["new browser"]
        assert "alternative" in new_browser
        assert "example" in new_browser
        assert "explanation" in new_browser
        assert new_browser["alternative"] == "Open Browser"

    def test_validate_keyword_for_selenium_session_rejects_browser_keyword(self):
        """Selenium session should reject Browser Library keywords."""
        plugin = SeleniumLibraryPlugin()

        # Create a session with Selenium preference
        session = ExecutionSession(session_id="selenium-test")
        session.explicit_library_preference = "SeleniumLibrary"

        # Validate a Browser Library keyword
        result = plugin.validate_keyword_for_session(
            session, "New Browser", "Browser"
        )

        assert result is not None
        assert result["success"] is False
        assert "Browser Library" in result["error"]
        assert "SeleniumLibrary" in result["error"]
        assert result["alternative"] == "Open Browser"

    def test_validate_keyword_for_selenium_session_accepts_selenium_keyword(self):
        """Selenium session should accept SeleniumLibrary keywords."""
        plugin = SeleniumLibraryPlugin()

        session = ExecutionSession(session_id="selenium-test")
        session.explicit_library_preference = "SeleniumLibrary"

        # Validate a SeleniumLibrary keyword - should return None (valid)
        result = plugin.validate_keyword_for_session(
            session, "Open Browser", "SeleniumLibrary"
        )

        assert result is None  # None means valid

    def test_validate_keyword_for_non_selenium_session_allows_any(self):
        """Non-Selenium session should not validate keywords."""
        plugin = SeleniumLibraryPlugin()

        # Create a session without Selenium preference
        session = ExecutionSession(session_id="other-test")
        session.explicit_library_preference = None

        # Should return None (no validation needed)
        result = plugin.validate_keyword_for_session(
            session, "New Browser", "Browser"
        )

        assert result is None


class TestPluginManagerValidation:
    """Test plugin manager validation methods."""

    def test_manager_get_incompatible_libraries_for_browser(self):
        """Manager should return incompatible libraries from Browser plugin."""
        # Ensure plugins are loaded
        library_registry.get_all_libraries()
        manager = get_library_plugin_manager()

        incompatible = manager.get_incompatible_libraries("Browser")
        assert "SeleniumLibrary" in incompatible

    def test_manager_get_incompatible_libraries_for_selenium(self):
        """Manager should return incompatible libraries from Selenium plugin."""
        library_registry.get_all_libraries()
        manager = get_library_plugin_manager()

        incompatible = manager.get_incompatible_libraries("SeleniumLibrary")
        assert "Browser" in incompatible

    def test_manager_get_keyword_alternatives_for_browser(self):
        """Manager should return keyword alternatives from Browser plugin."""
        library_registry.get_all_libraries()
        manager = get_library_plugin_manager()

        alternatives = manager.get_keyword_alternatives("Browser")
        assert "open browser" in alternatives
        assert alternatives["open browser"]["alternative"] == "New Browser + New Page"

    def test_manager_get_keyword_alternatives_for_selenium(self):
        """Manager should return keyword alternatives from Selenium plugin."""
        library_registry.get_all_libraries()
        manager = get_library_plugin_manager()

        alternatives = manager.get_keyword_alternatives("SeleniumLibrary")
        assert "new browser" in alternatives
        assert alternatives["new browser"]["alternative"] == "Open Browser"

    def test_manager_validate_keyword_for_session_browser(self):
        """Manager should delegate validation to Browser plugin."""
        library_registry.get_all_libraries()
        manager = get_library_plugin_manager()

        session = ExecutionSession(session_id="browser-test")
        session.explicit_library_preference = "Browser"

        # Use a Selenium-only keyword (not shared between libraries)
        result = manager.validate_keyword_for_session(
            "Browser", session, "Input Text", "SeleniumLibrary"
        )

        assert result is not None
        assert result["success"] is False

    def test_manager_validate_keyword_for_session_selenium(self):
        """Manager should delegate validation to Selenium plugin."""
        library_registry.get_all_libraries()
        manager = get_library_plugin_manager()

        session = ExecutionSession(session_id="selenium-test")
        session.explicit_library_preference = "SeleniumLibrary"

        result = manager.validate_keyword_for_session(
            "SeleniumLibrary", session, "New Browser", "Browser"
        )

        assert result is not None
        assert result["success"] is False

    def test_manager_validate_keyword_for_unknown_library(self):
        """Manager should return None for unknown libraries."""
        manager = get_library_plugin_manager()

        session = ExecutionSession(session_id="test")

        result = manager.validate_keyword_for_session(
            "UnknownLibrary", session, "SomeKeyword", "OtherLibrary"
        )

        assert result is None


class TestKeywordAlternativesContent:
    """Test that keyword alternatives contain useful information."""

    def test_browser_alternatives_have_complete_info(self):
        """Each Browser alternative should have all required fields."""
        plugin = BrowserLibraryPlugin()
        alternatives = plugin.get_keyword_alternatives()

        for keyword, info in alternatives.items():
            assert "alternative" in info, f"Missing 'alternative' for {keyword}"
            assert "example" in info, f"Missing 'example' for {keyword}"
            assert "explanation" in info, f"Missing 'explanation' for {keyword}"
            assert len(info["alternative"]) > 0, f"Empty 'alternative' for {keyword}"
            assert len(info["example"]) > 0, f"Empty 'example' for {keyword}"
            assert len(info["explanation"]) > 0, f"Empty 'explanation' for {keyword}"

    def test_selenium_alternatives_have_complete_info(self):
        """Each Selenium alternative should have all required fields."""
        plugin = SeleniumLibraryPlugin()
        alternatives = plugin.get_keyword_alternatives()

        for keyword, info in alternatives.items():
            assert "alternative" in info, f"Missing 'alternative' for {keyword}"
            assert "example" in info, f"Missing 'example' for {keyword}"
            assert "explanation" in info, f"Missing 'explanation' for {keyword}"
            assert len(info["alternative"]) > 0, f"Empty 'alternative' for {keyword}"
            assert len(info["example"]) > 0, f"Empty 'example' for {keyword}"
            assert len(info["explanation"]) > 0, f"Empty 'explanation' for {keyword}"
