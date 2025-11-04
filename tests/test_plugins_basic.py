"""Tests for the library plugin system integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest

from robotmcp.components.execution.page_source_service import PageSourceService
from robotmcp.components.execution.session_manager import SessionManager
from robotmcp.config import library_registry
from robotmcp.models.session_models import ExecutionSession
from robotmcp.plugins import (
    get_library_plugin_manager,
    reset_library_plugin_manager_for_tests,
)
from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryMetadata,
    LibraryPlugin,
    LibraryStateProvider,
)


@pytest.fixture(autouse=True)
def reset_plugin_state():
    """Reset plugin manager and registry state before and after each test."""
    reset_library_plugin_manager_for_tests()
    library_registry._reset_plugin_state_for_tests()  # type: ignore[attr-defined]
    yield
    reset_library_plugin_manager_for_tests()
    library_registry._reset_plugin_state_for_tests()  # type: ignore[attr-defined]


def _make_metadata(name: str) -> LibraryMetadata:
    return LibraryMetadata(
        name=name,
        package_name=f"{name.lower()}-package",
        import_path=f"{name}.Library",
        description=f"{name} plugin",
        library_type="external",
        use_cases=["testing"],
        categories=["web"],
        installation_command="pip install example",
        load_priority=42,
    )


def _make_capabilities() -> LibraryCapabilities:
    return LibraryCapabilities(
        contexts=["web"],
        requires_type_conversion=True,
        supports_async=True,
    )


def test_builtin_plugins_available_via_registry():
    libs = library_registry.get_all_libraries()
    assert "Browser" in libs
    assert libs["Browser"].import_path == "Browser"


def test_custom_plugin_registration_reflected_in_registry():
    manager = get_library_plugin_manager()
    custom_plugin = StaticLibraryPlugin(
        metadata=_make_metadata("CustomWeb"),
        capabilities=_make_capabilities(),
        install_actions=[InstallAction(description="Install", command=["pip install customweb"])]
    )
    manager.register_plugin(custom_plugin, source="test")

    libs = library_registry.get_all_libraries()
    assert "CustomWeb" in libs
    assert "CustomWeb" in library_registry.get_libraries_requiring_type_conversion()


class HookRecordingPlugin(StaticLibraryPlugin):
    def __init__(self, metadata: LibraryMetadata, capabilities: LibraryCapabilities):
        super().__init__(metadata=metadata, capabilities=capabilities)
        self.started: List[str] = []
        self.ended: List[str] = []

    def on_session_start(self, session: ExecutionSession) -> None:
        self.started.append(session.session_id)

    def on_session_end(self, session: ExecutionSession) -> None:
        self.ended.append(session.session_id)


def test_session_manager_invokes_plugin_hooks():
    manager = get_library_plugin_manager()
    plugin = HookRecordingPlugin(
        metadata=_make_metadata("HookLib"),
        capabilities=_make_capabilities(),
    )
    manager.register_plugin(plugin, source="test")

    session_manager = SessionManager()
    session = session_manager.create_session("session-hook-test")
    assert plugin.started == [session.session_id]

    session_manager.remove_session(session.session_id)
    assert plugin.ended == [session.session_id]


@dataclass
class DummyStateProvider:
    page_source_value: str

    async def get_page_source(
        self,
        session: ExecutionSession,
        *,
        full_source: bool,
        filtered: bool,
        filtering_level: str,
        include_reduced_dom: bool,
    ) -> Dict[str, str]:
        return {
            "success": True,
            "session_id": session.session_id,
            "page_source": self.page_source_value,
            "page_source_length": len(self.page_source_value),
            "current_url": "http://plugin.example",
            "page_title": "Plugin Page",
            "filtering_applied": filtered,
        }

    async def get_application_state(self, session: ExecutionSession) -> Optional[Dict[str, str]]:
        return {"state": "ok"}


class BrowserOverridePlugin(StaticLibraryPlugin):
    def __init__(self, provider: LibraryStateProvider):
        metadata = LibraryMetadata(
            name="Browser",
            package_name="robotframework-browser",
            import_path="Browser",
            description="Browser override for tests",
            library_type="external",
            categories=["web"],
            load_priority=5,
        )
        capabilities = LibraryCapabilities(contexts=["web"], supports_page_source=True)
        super().__init__(metadata=metadata, capabilities=capabilities)
        self._provider = provider

    def get_state_provider(self) -> LibraryStateProvider:
        return self._provider


@pytest.mark.asyncio
async def test_page_source_service_uses_plugin_provider():
    # Ensure built-in plugins load first, then override Browser with test plugin
    library_registry.get_all_libraries()
    provider = DummyStateProvider(page_source_value="<html>plugin</html>")
    override_plugin = BrowserOverridePlugin(provider)
    manager = get_library_plugin_manager()
    manager.register_plugin(override_plugin, source="test")

    session = ExecutionSession(session_id="plugin-session")
    session.imported_libraries.append("Browser")
    session.browser_state.active_library = "browser"

    service = PageSourceService()
    result = await service.get_page_source(
        session=session,
        browser_library_manager=None,
        full_source=True,
        filtered=False,
        filtering_level="standard",
        include_reduced_dom=False,
    )

    assert result["success"] is True
    assert result.get("page_source") == "<html>plugin</html>"
