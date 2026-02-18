"""Tests for PlatynUI library name normalization and placeholder detection.

Verifies the fixes from GNOME_CALCULATOR_TEST_REPORT.md analysis:
1. Library name normalization: PlatynUI → PlatynUI.BareMetal
2. LibraryCategory.DESKTOP enum added
3. PlatynUI categories include 'desktop'
4. _normalize_library_name() works for aliases and registry fallback
"""

from __future__ import annotations

import pytest


class TestLibraryCategoryDesktop:
    """Verify DESKTOP was added to LibraryCategory enum."""

    def test_desktop_category_exists(self):
        from robotmcp.config.library_registry import LibraryCategory

        assert hasattr(LibraryCategory, "DESKTOP")
        assert LibraryCategory.DESKTOP.value == "desktop"

    def test_all_expected_categories(self):
        from robotmcp.config.library_registry import LibraryCategory

        expected = {
            "core", "web", "api", "mobile", "desktop",
            "database", "data", "system", "network",
            "visual", "testing", "utilities",
        }
        actual = {c.value for c in LibraryCategory}
        assert expected == actual


class TestPlatynUIRegistryEntry:
    """Verify PlatynUI is correctly registered in the library registry."""

    def test_platynui_in_installation_info(self):
        from robotmcp.config.library_registry import get_installation_info

        info = get_installation_info()
        assert "PlatynUI" in info
        assert info["PlatynUI"]["import"] == "PlatynUI.BareMetal"
        assert info["PlatynUI"]["package"] == "robotframework-platynui"

    def test_platynui_in_recommendation_info(self):
        from robotmcp.config.library_registry import get_recommendation_info

        recs = get_recommendation_info()
        platynui = [r for r in recs if r["name"] == "PlatynUI"]
        assert len(platynui) == 1
        rec = platynui[0]
        assert "desktop testing" in rec["use_cases"]
        assert "desktop" in rec["categories"]

    def test_platynui_has_desktop_category(self):
        from robotmcp.config.library_registry import (
            LibraryCategory,
            get_libraries_by_category,
        )

        desktop_libs = get_libraries_by_category(LibraryCategory.DESKTOP)
        assert "PlatynUI" in desktop_libs

    def test_platynui_import_path_differs_from_name(self):
        from robotmcp.config.library_registry import get_library_config

        cfg = get_library_config("PlatynUI")
        assert cfg is not None
        assert cfg.import_path == "PlatynUI.BareMetal"
        assert cfg.name == "PlatynUI"


class TestNormalizeLibraryName:
    """Test the _normalize_library_name function."""

    def _normalize(self, name: str) -> str:
        # Import from server module
        from robotmcp.server import _normalize_library_name

        return _normalize_library_name(name)

    def test_platynui_normalizes_to_baremetal(self):
        assert self._normalize("PlatynUI") == "PlatynUI.BareMetal"

    def test_platynui_baremetal_unchanged(self):
        assert self._normalize("PlatynUI.BareMetal") == "PlatynUI.BareMetal"

    def test_builtin_unchanged(self):
        assert self._normalize("BuiltIn") == "BuiltIn"

    def test_browser_unchanged(self):
        assert self._normalize("Browser") == "Browser"

    def test_selenium_unchanged(self):
        assert self._normalize("SeleniumLibrary") == "SeleniumLibrary"

    def test_collections_unchanged(self):
        assert self._normalize("Collections") == "Collections"

    def test_empty_string_unchanged(self):
        assert self._normalize("") == ""

    def test_unknown_library_unchanged(self):
        assert self._normalize("SomeUnknownLibrary") == "SomeUnknownLibrary"


class TestLibraryNameAliases:
    """Test the _LIBRARY_NAME_ALIASES dict."""

    def test_alias_dict_has_platynui(self):
        from robotmcp.server import _LIBRARY_NAME_ALIASES

        assert "PlatynUI" in _LIBRARY_NAME_ALIASES
        assert _LIBRARY_NAME_ALIASES["PlatynUI"] == "PlatynUI.BareMetal"


class TestLibraryRecommenderDesktop:
    """Verify LibraryRecommender includes PlatynUI for desktop scenarios."""

    def test_desktop_category_weight(self):
        from robotmcp.components.library_recommender import LibraryRecommender

        assert "desktop" in LibraryRecommender.CATEGORY_WEIGHTS
        assert LibraryRecommender.CATEGORY_WEIGHTS["desktop"] > 0

    def test_recommend_for_desktop_scenario(self):
        from robotmcp.components.library_recommender import LibraryRecommender

        recommender = LibraryRecommender()
        result = recommender.recommend_libraries(
            "Desktop automation: automate GNOME calculator with PlatynUI",
            context="desktop",
        )
        assert result["success"]
        names = [r["library_name"] for r in result["recommendations"]]
        # PlatynUI should be in the recommendations for desktop scenarios
        assert "PlatynUI" in names

    def test_recommend_for_desktop_gui_automation(self):
        from robotmcp.components.library_recommender import LibraryRecommender

        recommender = LibraryRecommender()
        result = recommender.recommend_libraries(
            "Test native desktop GUI application",
            context="desktop",
        )
        assert result["success"]
        names = [r["library_name"] for r in result["recommendations"]]
        # PlatynUI should appear for generic desktop automation
        assert "PlatynUI" in names


class TestLibraryDetectionPlatynUI:
    """Verify library detection patterns score PlatynUI for desktop scenarios."""

    def test_platynui_explicit_mention(self):
        from robotmcp.utils.library_detection import LibraryDetector

        detector = LibraryDetector()
        scores = detector.get_scores("use PlatynUI to test desktop app")
        assert "PlatynUI" in scores
        assert scores["PlatynUI"] >= 9  # \bplatynui\b has weight 9

    def test_desktop_automation_matches_platynui(self):
        from robotmcp.utils.library_detection import LibraryDetector

        detector = LibraryDetector()
        scores = detector.get_scores("desktop automation testing")
        assert "PlatynUI" in scores
        assert scores["PlatynUI"] >= 5

    def test_baremetal_matches_platynui(self):
        from robotmcp.utils.library_detection import LibraryDetector

        detector = LibraryDetector()
        scores = detector.get_scores("use baremetal for GUI testing")
        assert "PlatynUI" in scores
        assert scores["PlatynUI"] >= 8

    def test_detect_returns_platynui_for_desktop(self):
        from robotmcp.utils.library_detection import LibraryDetector

        detector = LibraryDetector()
        result = detector.detect("automate desktop GUI with PlatynUI")
        assert result == "PlatynUI"

    def test_detect_all_includes_platynui(self):
        from robotmcp.utils.library_detection import LibraryDetector

        detector = LibraryDetector()
        results = detector.detect_all("PlatynUI desktop automation test")
        names = [name for name, score in results]
        assert "PlatynUI" in names
