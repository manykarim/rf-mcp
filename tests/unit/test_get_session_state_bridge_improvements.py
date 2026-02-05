"""Tests for get_session_state attach bridge improvements (S1-S5).

Covers the following improvements from docs/issues/get_session_state_attach_bridge_analysis.md:
  S1 - Enable ARIA snapshots via bridge
  S2 - Apply filtering to bridge page source
  S3 - Add bridge awareness to application_state
  S4 - Add bridge routing to libraries section
  S5 - Enrich summary section with bridge data
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test Helpers / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_bridge_client():
    """Create a mock ExternalRFClient for testing bridge routing.

    Mocks the dedicated bridge methods (get_page_source, get_aria_snapshot)
    as used by the Phase 3 implementation in _get_page_source_payload.
    """
    client = MagicMock()

    # Default diagnostics response
    client.diagnostics.return_value = {
        "success": True,
        "result": {
            "libraries": ["Browser", "Collections", "String"],
            "context": True,
        },
    }

    # Default get_variables response
    client.get_variables.return_value = {
        "success": True,
        "result": {
            "${BASE_URL}": "https://example.com",
            "${TIMEOUT}": "30",
        },
    }

    # Default get_page_source response (Phase 3: dedicated method instead of run_keyword)
    client.get_page_source.return_value = {
        "success": True,
        "result": "<html><head><title>Test Page</title></head><body><h1>Hello</h1></body></html>",
        "library": "Browser",
    }

    # Default get_aria_snapshot response (Phase 3: dedicated method instead of run_keyword)
    client.get_aria_snapshot.return_value = {
        "success": True,
        "result": "- heading: Hello\n- button: Submit",
        "format": "yaml",
        "selector": "css=html",
        "library": "Browser",
    }

    # Legacy run_keyword for backwards compatibility (used by some other tests)
    def run_keyword_side_effect(keyword, args=None, **kwargs):
        if keyword == "Get Page Source":
            return {
                "success": True,
                "result": "<html><head><title>Test Page</title></head><body><h1>Hello</h1></body></html>",
            }
        elif keyword == "Get Source":
            return {
                "success": True,
                "result": "<html><body>Selenium Source</body></html>",
            }
        elif keyword == "Get Aria Snapshot":
            return {
                "success": True,
                "result": "- heading: Hello\n- button: Submit",
            }
        return {"success": False, "error": f"Unknown keyword: {keyword}"}

    client.run_keyword.side_effect = run_keyword_side_effect

    return client


# =========================================================================
# Test Group 1 -- S1: ARIA Snapshots via Bridge
# =========================================================================


class TestAriaSnapshotsViaBridge:
    """S1: ARIA snapshots should be retrieved via run_keyword instead of being hardcoded as unsupported."""

    @pytest.mark.asyncio
    async def test_aria_snapshot_retrieved_via_bridge(self, mock_bridge_client, monkeypatch):
        """When bridge is active and include_reduced_dom=True, ARIA snapshot is fetched via run_keyword."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            include_reduced_dom=True,
        )

        assert result["success"] is True
        assert result["source"] == "attach_bridge"
        assert result["aria_snapshot"]["success"] is True
        assert result["aria_snapshot"]["content"] == "- heading: Hello\n- button: Submit"
        assert result["aria_snapshot"]["format"] == "yaml"
        assert result["aria_snapshot"]["source"] == "attach_bridge"

        # Verify get_aria_snapshot was called (Phase 3: dedicated method)
        mock_bridge_client.get_aria_snapshot.assert_called_once_with(
            selector="css=html",
            format_type="yaml",
        )

    @pytest.mark.asyncio
    async def test_aria_snapshot_skipped_when_not_requested(self, mock_bridge_client, monkeypatch):
        """When include_reduced_dom=False, ARIA snapshot should be skipped."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            include_reduced_dom=False,
        )

        assert result["success"] is True
        assert result["aria_snapshot"]["skipped"] is True
        assert "content" not in result["aria_snapshot"]

    @pytest.mark.asyncio
    async def test_aria_snapshot_failure_handled_gracefully(self, mock_bridge_client, monkeypatch):
        """When ARIA snapshot fails, the error is captured but page source still returns."""
        # Override get_page_source for this test
        mock_bridge_client.get_page_source.return_value = {
            "success": True,
            "result": "<html></html>",
            "library": "Browser",
        }
        # Override get_aria_snapshot to fail
        mock_bridge_client.get_aria_snapshot.return_value = {
            "success": False,
            "error": "Browser not ready",
        }

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            include_reduced_dom=True,
        )

        assert result["success"] is True
        assert result["page_source"] == "<html></html>"
        assert result["aria_snapshot"]["success"] is False
        assert "error" in result["aria_snapshot"]


# =========================================================================
# Test Group 2 -- S2: Page Source Filtering
# =========================================================================


class TestPageSourceFiltering:
    """S2: Page source filtering should be applied to bridge-returned HTML."""

    @pytest.mark.asyncio
    async def test_filtering_applied_when_requested(self, mock_bridge_client, monkeypatch):
        """When filtered=True, page source should be filtered locally."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock the filter function
        mock_filter = MagicMock(return_value="<html>Filtered Content</html>")
        monkeypatch.setattr(
            "robotmcp.components.execution.page_source_service.PageSourceService.filter_page_source",
            mock_filter,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            filtered=True,
            filtering_level="aggressive",
        )

        assert result["success"] is True
        assert result["metadata"]["filtered"] is True
        assert result["metadata"]["full"] is False
        mock_filter.assert_called_once()
        # Verify filtering_level was passed
        assert mock_filter.call_args[0][1] == "aggressive"

    @pytest.mark.asyncio
    async def test_no_filtering_when_not_requested(self, mock_bridge_client, monkeypatch):
        """When filtered=False, page source should not be filtered."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            filtered=False,
        )

        assert result["success"] is True
        assert result["metadata"]["filtered"] is False
        assert result["metadata"]["full"] is True


# =========================================================================
# Test Group 3 -- S3: Application State Bridge Awareness
# =========================================================================


class TestApplicationStateBridgeAwareness:
    """S3: application_state section should route through bridge when available."""

    @pytest.mark.asyncio
    async def test_application_state_uses_bridge_for_dom(self, mock_bridge_client, monkeypatch):
        """When bridge is active, DOM state should come from bridge page source."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_application_state_payload

        result = await _get_application_state_payload(
            state_type="dom",
            session_id="test-session",
        )

        assert result["success"] is True
        assert result["source"] == "attach_bridge"
        assert "dom_state" in result
        assert result["dom_state"]["success"] is True
        assert "page_source" in result["dom_state"]

    @pytest.mark.asyncio
    async def test_application_state_includes_variables(self, mock_bridge_client, monkeypatch):
        """When bridge is active, variables should be included from bridge."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_application_state_payload

        result = await _get_application_state_payload(
            state_type="all",
            session_id="test-session",
        )

        assert result["success"] is True
        assert "variables" in result
        assert "BASE_URL" in result["variables"]
        assert result["variable_count"] == 2

    @pytest.mark.asyncio
    async def test_application_state_api_and_database_not_available(
        self, mock_bridge_client, monkeypatch
    ):
        """API and database states should indicate they're not available via bridge."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_application_state_payload

        result = await _get_application_state_payload(
            state_type="all",
            session_id="test-session",
        )

        assert result["api_state"]["success"] is False
        assert "not available" in result["api_state"]["note"]
        assert result["database_state"]["success"] is False
        assert "not available" in result["database_state"]["note"]


# =========================================================================
# Test Group 4 -- S4: Libraries Bridge Routing
# =========================================================================


class TestLibrariesBridgeRouting:
    """S4: libraries section should route through bridge diagnostics."""

    @pytest.mark.asyncio
    async def test_libraries_retrieved_from_bridge(self, mock_bridge_client, monkeypatch):
        """When bridge is active, libraries should come from diagnostics."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_loaded_libraries_payload

        result = await _get_loaded_libraries_payload()

        assert result["success"] is True
        assert result["source"] == "attach_bridge"
        assert result["libraries"] == ["Browser", "Collections", "String"]
        assert result["library_count"] == 3
        assert result["context_active"] is True

        mock_bridge_client.diagnostics.assert_called_once()

    @pytest.mark.asyncio
    async def test_libraries_fallback_on_bridge_failure(self, mock_bridge_client, monkeypatch):
        """When bridge diagnostics fails, should fall back to local."""
        mock_bridge_client.diagnostics.return_value = {"success": False, "error": "not reachable"}

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock the local fallback
        mock_local_status = {
            "success": True,
            "source": "local",
            "libraries": ["BuiltIn"],
        }
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.get_library_status",
            lambda: mock_local_status,
        )

        from robotmcp.server import _get_loaded_libraries_payload

        result = await _get_loaded_libraries_payload()

        assert result["source"] == "local"


# =========================================================================
# Test Group 5 -- S5: Summary Section Enrichment
# =========================================================================


class TestSummaryBridgeEnrichment:
    """S5: summary section should be enriched with bridge diagnostics."""

    @pytest.mark.asyncio
    async def test_summary_enriched_with_bridge_data(self, mock_bridge_client, monkeypatch):
        """When bridge is active, summary should include attach_bridge info."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock session manager to return a session
        mock_session = MagicMock()
        mock_session.get_session_info.return_value = {
            "session_id": "test-session",
            "step_count": 5,
        }
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: mock_session,
        )

        from robotmcp.server import _get_session_info_payload

        result = await _get_session_info_payload("test-session")

        assert result["success"] is True
        assert "attach_bridge" in result["session_info"]
        assert result["session_info"]["attach_bridge"]["active"] is True
        assert result["session_info"]["attach_bridge"]["libraries"] == [
            "Browser",
            "Collections",
            "String",
        ]
        assert result["session_info"]["attach_bridge"]["context_active"] is True

    @pytest.mark.asyncio
    async def test_summary_without_session_but_with_bridge(self, mock_bridge_client, monkeypatch):
        """When session doesn't exist but bridge is active, should still return success with bridge info."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock session manager to return None (no session)
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: None,
        )

        from robotmcp.server import _get_session_info_payload

        result = await _get_session_info_payload("nonexistent-session")

        # Should succeed because bridge is active
        assert result["success"] is True
        assert "attach_bridge" in result["session_info"]
        assert result["session_info"]["attach_bridge"]["active"] is True

    @pytest.mark.asyncio
    async def test_summary_fails_without_session_and_bridge(self, monkeypatch):
        """When neither session nor bridge exists, should return error."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: None,
        )

        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: None,
        )
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_all_session_ids",
            lambda: ["session-1", "session-2"],
        )

        from robotmcp.server import _get_session_info_payload

        result = await _get_session_info_payload("nonexistent-session")

        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["available_sessions"] == ["session-1", "session-2"]


# =========================================================================
# Test Group 6 -- Integration Tests
# =========================================================================


class TestGetSessionStateIntegration:
    """Integration tests for get_session_state with bridge routing."""

    @pytest.mark.asyncio
    async def test_all_sections_use_bridge_when_configured(self, mock_bridge_client, monkeypatch):
        """When bridge is active, all supported sections should route through it."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock session
        mock_session = MagicMock()
        mock_session.get_session_info.return_value = {"session_id": "test"}
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: mock_session,
        )

        # Mock instruction hooks
        monkeypatch.setattr(
            "robotmcp.server._track_tool_result",
            lambda *a, **kw: None,
        )

        from robotmcp.server import get_session_state

        # Access the underlying function
        get_state_fn = get_session_state.fn

        result = await get_state_fn(
            session_id="test",
            sections=["summary", "page_source", "variables", "libraries", "application_state"],
        )

        assert result["success"] is True

        # Check each section used bridge
        sections = result["sections"]

        # summary should have attach_bridge info
        assert "attach_bridge" in sections["summary"]["session_info"]
        assert sections["summary"]["session_info"]["attach_bridge"]["active"] is True

        # page_source should come from bridge
        assert sections["page_source"]["source"] == "attach_bridge"

        # variables should come from bridge
        assert sections["variables"]["source"] == "attach_bridge"

        # libraries should come from bridge
        assert sections["libraries"]["source"] == "attach_bridge"

        # application_state should come from bridge
        assert sections["application_state"]["source"] == "attach_bridge"
