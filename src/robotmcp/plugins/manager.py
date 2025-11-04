"""Plugin manager responsible for registering and serving library plugins."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Iterable, List, Optional

from .contracts import (
    InstallAction,
    KeywordOverrideHandler,
    LibraryCapabilities,
    LibraryHints,
    LibraryPlugin,
    LibraryStateProvider,
    LibraryMetadata,
    PromptBundle,
    TypeConversionProvider,
)
from .discovery import iter_entry_point_plugins, iter_manifest_plugins

logger = logging.getLogger(__name__)


class LibraryPluginManager:
    """Central registry of rf-mcp library plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, LibraryPlugin] = {}
        self._metadata: Dict[str, LibraryMetadata] = {}
        self._capabilities: Dict[str, LibraryCapabilities] = {}
        self._install_actions: Dict[str, List[InstallAction]] = {}
        self._hints: Dict[str, LibraryHints] = {}
        self._prompts: Dict[str, PromptBundle] = {}
        self._lock = threading.RLock()
        self._sources: Dict[str, str] = {}
        self._keyword_map: Dict[str, str] = {}
        self._keyword_overrides: Dict[tuple[str, str], KeywordOverrideHandler] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_plugin(self, plugin: LibraryPlugin, source: str = "builtin") -> None:
        """Register a plugin instance."""
        metadata = plugin.get_metadata()

        if metadata.schema_version < 1:
            logger.warning(
                "Skipping plugin %s due to unsupported schema version %s",
                metadata.name,
                metadata.schema_version,
            )
            return

        with self._lock:
            self._plugins[metadata.name] = plugin
            self._metadata[metadata.name] = metadata

            capabilities = plugin.get_capabilities()
            if capabilities:
                self._capabilities[metadata.name] = capabilities
            else:
                self._capabilities.pop(metadata.name, None)

            install_actions = plugin.get_install_actions()
            if install_actions:
                self._install_actions[metadata.name] = list(install_actions)
            else:
                self._install_actions.pop(metadata.name, None)

            hints = plugin.get_hints()
            if hints:
                self._hints[metadata.name] = hints
            else:
                self._hints.pop(metadata.name, None)

            prompts = plugin.get_prompt_bundle()
            if prompts:
                self._prompts[metadata.name] = prompts
            else:
                self._prompts.pop(metadata.name, None)

            self._sources[metadata.name] = source

            keyword_map = plugin.get_keyword_library_map()
            if keyword_map:
                for keyword, lib_name in keyword_map.items():
                    if not keyword:
                        continue
                    self._keyword_map[keyword.lower()] = lib_name

            overrides = plugin.get_keyword_overrides()
            if overrides:
                for keyword, handler in overrides.items():
                    if not keyword or handler is None:
                        continue
                    self._keyword_overrides[(metadata.name, keyword.lower())] = handler

            logger.debug("Registered library plugin '%s' from %s", metadata.name, source)

    def register_plugins(
        self, plugins: Iterable[LibraryPlugin], source: str = "builtin"
    ) -> None:
        for plugin in plugins:
            self.register_plugin(plugin, source=source)

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def discover_entry_point_plugins(self) -> None:
        self.register_plugins(iter_entry_point_plugins(), source="entry_point")

    def discover_manifest_plugins(self, paths: Optional[List[str]] = None) -> None:
        self.register_plugins(iter_manifest_plugins(paths), source="manifest")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get_plugin(self, name: str) -> Optional[LibraryPlugin]:
        return self._plugins.get(name)

    def get_metadata(self, name: str) -> Optional[LibraryMetadata]:
        return self._metadata.get(name)

    def get_capabilities(self, name: str) -> Optional[LibraryCapabilities]:
        return self._capabilities.get(name)

    def get_install_actions(self, name: str) -> List[InstallAction]:
        return self._install_actions.get(name, [])

    def get_hints(self, name: str) -> Optional[LibraryHints]:
        return self._hints.get(name)

    def get_prompt_bundle(self, name: str) -> Optional[PromptBundle]:
        return self._prompts.get(name)

    def get_plugin_source(self, name: str) -> Optional[str]:
        return self._sources.get(name)

    def get_state_provider(self, name: str) -> Optional[LibraryStateProvider]:
        plugin = self.get_plugin(name)
        if plugin:
            try:
                return plugin.get_state_provider()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to obtain state provider for %s: %s", name, exc
                )
        return None

    def get_type_conversion_provider(self, name: str) -> Optional[TypeConversionProvider]:
        plugin = self.get_plugin(name)
        if plugin:
            try:
                return plugin.get_type_converters()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to obtain type conversion provider for %s: %s", name, exc
                )
        return None

    def list_plugin_names(self) -> List[str]:
        return sorted(self._metadata.keys(), key=str.casefold)

    def iter_metadata(self):
        for name, meta in self._metadata.items():
            yield name, meta

    def iter_capabilities(self):
        for name, cap in self._capabilities.items():
            yield name, cap

    def get_library_for_keyword(self, keyword: str) -> Optional[str]:
        return self._keyword_map.get(keyword.lower())

    def get_keyword_override(
        self, library_name: Optional[str], keyword: str
    ) -> Optional[KeywordOverrideHandler]:
        if not library_name:
            return None
        return self._keyword_overrides.get((library_name, keyword.lower()))

    def run_before_keyword_execution(
        self,
        library_name: Optional[str],
        session: Any,
        keyword_name: str,
        library_manager: Any,
        keyword_discovery: Any,
    ) -> None:
        if not library_name:
            return
        plugin = self._plugins.get(library_name)
        if not plugin:
            return
        try:
            plugin.before_keyword_execution(
                session,
                keyword_name,
                library_manager,
                keyword_discovery,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Plugin %s before_keyword_execution failed: %s",
                library_name,
                exc,
            )


_GLOBAL_MANAGER: Optional[LibraryPluginManager] = None
_GLOBAL_MANAGER_LOCK = threading.Lock()


def get_library_plugin_manager() -> LibraryPluginManager:
    """Return the process-wide plugin manager."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        with _GLOBAL_MANAGER_LOCK:
            if _GLOBAL_MANAGER is None:
                _GLOBAL_MANAGER = LibraryPluginManager()
    return _GLOBAL_MANAGER


def reset_library_plugin_manager_for_tests() -> None:
    """Reset the global manager (intended for tests)."""
    global _GLOBAL_MANAGER
    with _GLOBAL_MANAGER_LOCK:
        _GLOBAL_MANAGER = LibraryPluginManager()
