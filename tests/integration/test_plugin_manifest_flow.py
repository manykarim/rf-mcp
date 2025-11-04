"""End-to-end test verifying manifest-based plugin discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotmcp.components.execution.page_source_service import PageSourceService
from robotmcp.components.execution.session_manager import SessionManager
from robotmcp.config import library_registry
from robotmcp.plugins import get_library_plugin_manager


@pytest.mark.asyncio
async def test_manifest_plugin_flow(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "example_calculator.json"
    manifest.write_text(
        json.dumps(
            {
                "module": "examples.plugins.sample_plugin",
                "class": "ExampleCalculatorPlugin",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ROBOTMCP_PLUGIN_PATHS", str(manifest))
    library_registry._reset_plugin_state_for_tests()  # type: ignore[attr-defined]

    libs = library_registry.get_all_libraries()
    assert "ExampleCalculatorLibrary" in libs

    manager = get_library_plugin_manager()
    plugin = manager.get_plugin("ExampleCalculatorLibrary")
    assert plugin is not None
    assert manager.get_plugin_source("ExampleCalculatorLibrary") == "manifest"

    provider = plugin.get_state_provider()
    assert provider is not None

    session_manager = SessionManager()
    session = session_manager.create_session("calculator-demo")
    assert session.variables.get("CALC_LAST_RESULT") == "0"

    session.browser_state.active_library = "ExampleCalculatorLibrary"
    service = PageSourceService()
    result = await service.get_page_source(
        session=session,
        browser_library_manager=None,
        full_source=True,
        filtered=False,
        filtering_level="standard",
        include_reduced_dom=False,
    )

    assert result.get("success") is True
    assert "Calculator" in result.get("page_title", "")
