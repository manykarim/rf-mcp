"""Tests for P13: Library-aware keyword routing in PageSourceService.

Verifies that get_session_state uses the correct keywords for each library:
  - Browser Library: Get Page Source, Get Url, Get Title
  - SeleniumLibrary: Get Source, Get Location, Get Title
  - AppiumLibrary:   Get Source, Get Window Url, Get Window Title

Run with: uv run pytest tests/unit/test_page_source_keyword_routing.py -v
"""

__test__ = True

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from robotmcp.components.execution.page_source_service import PageSourceService
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession


@pytest.fixture
def service():
    return PageSourceService(config=ExecutionConfig())


def _make_session(imported_libraries: list) -> MagicMock:
    session = MagicMock(spec=ExecutionSession)
    session.session_id = "test_kw_routing"
    session.imported_libraries = imported_libraries
    session.browser_state = MagicMock()
    session.browser_state.page_source = ""
    session.browser_state.current_url = None
    session.browser_state.page_title = None
    session.variables = {}
    return session


# ---------------------------------------------------------------------------
# _keyword_candidates — unit tests for the routing helper
# ---------------------------------------------------------------------------


class TestKeywordCandidates:
    """Unit tests for PageSourceService._keyword_candidates()."""

    # ── Source keywords ──

    def test_browser_only_source(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["Browser", "BuiltIn"]
        )
        assert result == ("Get Page Source",)

    def test_selenium_only_source(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["SeleniumLibrary", "BuiltIn"]
        )
        assert result == ("Get Source",)

    def test_appium_only_source(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["AppiumLibrary", "BuiltIn"]
        )
        assert result == ("Get Source",)

    def test_browser_and_selenium_source(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["Browser", "SeleniumLibrary"]
        )
        assert result == ("Get Page Source", "Get Source")

    def test_browser_and_appium_source(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["Browser", "AppiumLibrary"]
        )
        # Both "Get Page Source" and "Get Source" — deduplicated
        assert result == ("Get Page Source", "Get Source")

    def test_selenium_and_appium_source(self):
        """SeleniumLibrary and AppiumLibrary both use 'Get Source' — no duplicate."""
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["SeleniumLibrary", "AppiumLibrary"]
        )
        assert result == ("Get Source",)

    def test_no_matching_library_falls_back(self):
        """When no library matches, return all unique keywords from the mapping."""
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["BuiltIn", "Collections"]
        )
        # Fallback: all unique keywords from the mapping
        assert "Get Page Source" in result
        assert "Get Source" in result

    def test_empty_imported_falls_back(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, []
        )
        assert len(result) >= 2  # At least the distinct keywords

    # ── URL keywords ──

    def test_browser_only_url(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["Browser"]
        )
        assert result == ("Get Url",)

    def test_selenium_only_url(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["SeleniumLibrary"]
        )
        assert result == ("Get Location",)

    def test_appium_only_url(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["AppiumLibrary"]
        )
        assert result == ("Get Window Url",)

    def test_browser_and_appium_url(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["Browser", "AppiumLibrary"]
        )
        assert result == ("Get Url", "Get Window Url")

    def test_selenium_and_appium_url(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["SeleniumLibrary", "AppiumLibrary"]
        )
        assert result == ("Get Location", "Get Window Url")

    # ── Title keywords ──

    def test_browser_only_title(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["Browser"]
        )
        assert result == ("Get Title",)

    def test_selenium_only_title(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["SeleniumLibrary"]
        )
        assert result == ("Get Title",)

    def test_appium_only_title(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["AppiumLibrary"]
        )
        assert result == ("Get Window Title",)

    def test_browser_and_selenium_title(self):
        """Browser and SeleniumLibrary both use 'Get Title' — deduplicated."""
        result = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["Browser", "SeleniumLibrary"]
        )
        assert result == ("Get Title",)

    def test_browser_and_appium_title(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["Browser", "AppiumLibrary"]
        )
        assert result == ("Get Title", "Get Window Title")

    def test_selenium_and_appium_title(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["SeleniumLibrary", "AppiumLibrary"]
        )
        assert result == ("Get Title", "Get Window Title")

    # ── Ordering: imported order preserved ──

    def test_import_order_preserved_source(self):
        """Candidates follow the order libraries appear in imported_libraries."""
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS,
            ["SeleniumLibrary", "Browser"],
        )
        assert result == ("Get Source", "Get Page Source")

    def test_import_order_preserved_url(self):
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS,
            ["AppiumLibrary", "Browser"],
        )
        assert result == ("Get Window Url", "Get Url")


# ---------------------------------------------------------------------------
# _get_page_source_via_rf_context — integration tests with mocked RF manager
# ---------------------------------------------------------------------------


class TestPageSourceViaRfContext:
    """Verify that _get_page_source_via_rf_context calls the right keywords."""

    def _run_with_tracking(self, service, session, success_kw=None):
        """Run _get_page_source_via_rf_context and track which keywords were called."""
        called_keywords = []

        def fake_execute(session_id, keyword_name, arguments, assign_to=None, session_variables=None):
            called_keywords.append(keyword_name)
            if keyword_name == success_kw:
                return {"success": True, "output": f"<html>mock from {keyword_name}</html>"}
            return {"success": False, "error": "Not found"}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_get_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_execute)
            mock_get_mgr.return_value = mgr
            result = service._get_page_source_via_rf_context(session)

        return result, called_keywords

    def test_selenium_only_calls_get_source(self, service):
        session = _make_session(["SeleniumLibrary", "BuiltIn"])
        result, called = self._run_with_tracking(service, session, "Get Source")
        assert "Get Source" in called
        assert "Get Page Source" not in called
        assert result == "<html>mock from Get Source</html>"

    def test_selenium_only_calls_get_location(self, service):
        session = _make_session(["SeleniumLibrary", "BuiltIn"])
        _, called = self._run_with_tracking(service, session, "Get Location")
        assert "Get Location" in called
        assert "Get Url" not in called

    def test_selenium_only_calls_get_title(self, service):
        session = _make_session(["SeleniumLibrary", "BuiltIn"])
        _, called = self._run_with_tracking(service, session, "Get Title")
        assert "Get Title" in called
        assert "Get Window Title" not in called

    def test_browser_only_calls_get_page_source(self, service):
        session = _make_session(["Browser", "BuiltIn"])
        result, called = self._run_with_tracking(service, session, "Get Page Source")
        assert "Get Page Source" in called
        assert "Get Source" not in called
        assert result == "<html>mock from Get Page Source</html>"

    def test_browser_only_calls_get_url(self, service):
        session = _make_session(["Browser", "BuiltIn"])
        _, called = self._run_with_tracking(service, session, "Get Url")
        assert "Get Url" in called
        assert "Get Location" not in called
        assert "Get Window Url" not in called

    def test_appium_only_calls_get_source(self, service):
        session = _make_session(["AppiumLibrary", "BuiltIn"])
        result, called = self._run_with_tracking(service, session, "Get Source")
        assert "Get Source" in called
        assert "Get Page Source" not in called
        assert result == "<html>mock from Get Source</html>"

    def test_appium_only_calls_get_window_url(self, service):
        session = _make_session(["AppiumLibrary", "BuiltIn"])
        _, called = self._run_with_tracking(service, session, "Get Window Url")
        assert "Get Window Url" in called
        assert "Get Url" not in called
        assert "Get Location" not in called

    def test_appium_only_calls_get_window_title(self, service):
        session = _make_session(["AppiumLibrary", "BuiltIn"])
        _, called = self._run_with_tracking(service, session, "Get Window Title")
        assert "Get Window Title" in called
        assert "Get Title" not in called

    def test_both_browser_and_selenium_tries_both_sources(self, service):
        """When both libraries imported, tries both source keywords."""
        session = _make_session(["Browser", "SeleniumLibrary", "BuiltIn"])
        result, called = self._run_with_tracking(service, session, "Get Source")
        # Should try Get Page Source first (Browser order), then Get Source
        assert "Get Page Source" in called
        assert "Get Source" in called

    def test_no_web_library_falls_back(self, service):
        """When no web library is imported, falls back to trying all keywords."""
        session = _make_session(["BuiltIn", "Collections"])
        _, called = self._run_with_tracking(service, session, None)
        # Should attempt fallback keywords
        source_calls = [k for k in called if "Source" in k or "source" in k.lower()]
        assert len(source_calls) >= 1


# ---------------------------------------------------------------------------
# _get_current_url — async tests
# ---------------------------------------------------------------------------


class TestGetCurrentUrl:
    """Verify _get_current_url calls the correct keyword per library."""

    @pytest.mark.asyncio
    async def test_selenium_uses_get_location(self, service):
        session = _make_session(["SeleniumLibrary"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Location":
                return {"success": True, "output": "https://example.com/selenium"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_current_url(session, None)

        assert result == "https://example.com/selenium"
        assert "Get Location" in called
        assert "Get Url" not in called

    @pytest.mark.asyncio
    async def test_appium_uses_get_window_url(self, service):
        session = _make_session(["AppiumLibrary"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Window Url":
                return {"success": True, "output": "https://example.com/appium"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_current_url(session, None)

        assert result == "https://example.com/appium"
        assert "Get Window Url" in called
        assert "Get Url" not in called
        assert "Get Location" not in called

    @pytest.mark.asyncio
    async def test_browser_uses_get_url(self, service):
        session = _make_session(["Browser"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Url":
                return {"success": True, "output": "https://example.com/browser"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_current_url(session, None)

        assert result == "https://example.com/browser"
        assert "Get Url" in called


# ---------------------------------------------------------------------------
# _get_page_title — async tests
# ---------------------------------------------------------------------------


class TestGetPageTitle:
    """Verify _get_page_title calls the correct keyword per library."""

    @pytest.mark.asyncio
    async def test_appium_uses_get_window_title(self, service):
        session = _make_session(["AppiumLibrary"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Window Title":
                return {"success": True, "output": "My App"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_page_title(session, None)

        assert result == "My App"
        assert "Get Window Title" in called
        assert "Get Title" not in called

    @pytest.mark.asyncio
    async def test_selenium_uses_get_title(self, service):
        session = _make_session(["SeleniumLibrary"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Title":
                return {"success": True, "output": "Selenium Page"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_page_title(session, None)

        assert result == "Selenium Page"
        assert "Get Title" in called
        assert "Get Window Title" not in called

    @pytest.mark.asyncio
    async def test_browser_uses_get_title(self, service):
        session = _make_session(["Browser"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Title":
                return {"success": True, "output": "Browser Page"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_page_title(session, None)

        assert result == "Browser Page"
        assert "Get Title" in called


# ---------------------------------------------------------------------------
# _get_page_source_unified_async — async tests
# ---------------------------------------------------------------------------


class TestPageSourceUnifiedAsync:
    """Verify _get_page_source_unified_async uses correct source keywords."""

    @pytest.mark.asyncio
    async def test_appium_uses_get_source(self, service):
        session = _make_session(["AppiumLibrary"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, assign_to=None, session_variables=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Source":
                return {"success": True, "output": "<xml>appium source</xml>"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_page_source_unified_async(session, None)

        assert result == "<xml>appium source</xml>"
        assert "Get Source" in called
        assert "Get Page Source" not in called

    @pytest.mark.asyncio
    async def test_browser_uses_get_page_source(self, service):
        session = _make_session(["Browser"])
        called = []

        def fake_exec(session_id, keyword_name, arguments=None, assign_to=None, session_variables=None, **kwargs):
            called.append(keyword_name)
            if keyword_name == "Get Page Source":
                return {"success": True, "output": "<html>browser source</html>"}
            return {"success": False}

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(side_effect=fake_exec)
            mock_mgr.return_value = mgr
            result = await service._get_page_source_unified_async(session, None)

        assert result == "<html>browser source</html>"
        assert "Get Page Source" in called
        assert "Get Source" not in called


# ---------------------------------------------------------------------------
# Edge cases and regression
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for keyword routing."""

    def test_all_three_libraries_imported(self):
        """Unlikely scenario: all three web/mobile libraries imported."""
        result = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS,
            ["Browser", "SeleniumLibrary", "AppiumLibrary"],
        )
        assert result == ("Get Url", "Get Location", "Get Window Url")

    def test_appium_and_builtin_only(self):
        """Common mobile scenario: AppiumLibrary + BuiltIn."""
        src = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["AppiumLibrary", "BuiltIn"]
        )
        url = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["AppiumLibrary", "BuiltIn"]
        )
        title = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["AppiumLibrary", "BuiltIn"]
        )
        assert src == ("Get Source",)
        assert url == ("Get Window Url",)
        assert title == ("Get Window Title",)

    def test_mapping_dicts_are_class_attributes(self):
        """Verify mappings are accessible as class attributes."""
        assert hasattr(PageSourceService, "_SOURCE_KEYWORDS")
        assert hasattr(PageSourceService, "_URL_KEYWORDS")
        assert hasattr(PageSourceService, "_TITLE_KEYWORDS")
        assert isinstance(PageSourceService._SOURCE_KEYWORDS, dict)
        assert isinstance(PageSourceService._URL_KEYWORDS, dict)
        assert isinstance(PageSourceService._TITLE_KEYWORDS, dict)

    def test_all_mappings_have_same_libraries(self):
        """All three mappings should cover the same set of libraries."""
        source_libs = set(PageSourceService._SOURCE_KEYWORDS.keys())
        url_libs = set(PageSourceService._URL_KEYWORDS.keys())
        title_libs = set(PageSourceService._TITLE_KEYWORDS.keys())
        assert source_libs == url_libs == title_libs

    def test_keyword_candidates_is_static(self):
        """_keyword_candidates should work without an instance."""
        result = PageSourceService._keyword_candidates(
            {"Browser": "Get Url"}, ["Browser"]
        )
        assert result == ("Get Url",)

    def test_duplicate_libraries_in_imported(self):
        """Duplicate entries in imported_libraries should not produce duplicate candidates."""
        result = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS,
            ["SeleniumLibrary", "SeleniumLibrary", "BuiltIn"],
        )
        assert result == ("Get Source",)

    def test_none_imported_libraries(self, service):
        """session.imported_libraries being None should not crash."""
        session = _make_session([])
        session.imported_libraries = None
        with patch(
            "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager"
        ) as mock_mgr:
            mgr = MagicMock()
            mgr.execute_keyword_with_context = MagicMock(return_value={"success": False})
            mock_mgr.return_value = mgr
            # Should not raise — falls back to trying all keywords
            service._get_page_source_via_rf_context(session)
