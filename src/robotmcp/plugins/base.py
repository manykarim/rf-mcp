"""Base plugin implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryHints,
    LibraryMetadata,
    LibraryPlugin,
    LibraryStateProvider,
    PromptBundle,
    TypeConversionProvider,
)


@dataclass
class StaticLibraryPlugin(LibraryPlugin):
    """Simple plugin backed by static metadata."""

    metadata: LibraryMetadata
    capabilities: Optional[LibraryCapabilities] = None
    install_actions: Optional[List[InstallAction]] = None
    hints: Optional[LibraryHints] = None
    prompt_bundle: Optional[PromptBundle] = None

    schema_version: int = 1

    def get_metadata(self) -> LibraryMetadata:
        return self.metadata

    def get_capabilities(self) -> Optional[LibraryCapabilities]:
        return self.capabilities

    def get_install_actions(self) -> Optional[List[InstallAction]]:
        return self.install_actions

    def get_hints(self) -> Optional[LibraryHints]:
        return self.hints

    def get_prompt_bundle(self) -> Optional[PromptBundle]:
        return self.prompt_bundle

    def get_state_provider(self) -> Optional[LibraryStateProvider]:
        return None

    def get_type_converters(self) -> Optional[TypeConversionProvider]:
        return None

    def get_keyword_library_map(self) -> Optional[Dict[str, str]]:  # type: ignore[override]
        return None

    def get_keyword_overrides(self):  # type: ignore[override]
        return None

    def before_keyword_execution(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        library_manager,
        keyword_discovery,
    ) -> None:  # type: ignore[override]
        return None

    def on_session_start(self, session: "ExecutionSession") -> None:
        return None

    def on_session_end(self, session: "ExecutionSession") -> None:
        return None


class ManifestLibraryPlugin(StaticLibraryPlugin):
    """Plugin constructed from manifest metadata."""

    # Inherits behaviour from StaticLibraryPlugin
    pass


# Avoid circular imports for typing
try:  # pragma: no cover - typing guard
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
