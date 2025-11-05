"""Contracts and dataclasses for rf-mcp library plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
    Callable,
    Awaitable,
)


@dataclass(frozen=True)
class InstallAction:
    """Represents a single installation step for a library."""

    description: str
    command: Sequence[str]


@dataclass(frozen=True)
class LibraryHints:
    """Optional guidance and examples provided by the plugin."""

    standard_keywords: List[str] = field(default_factory=list)
    error_hints: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptBundle:
    """Prompt snippets that can be injected into sampling or recommender flows."""

    recommendation: Optional[str] = None
    troubleshooting: Optional[str] = None
    sampling_notes: Optional[str] = None


@dataclass(frozen=True)
class LibraryCapabilities:
    """Capabilities exposed by the plugin to augment rf-mcp behaviour."""

    contexts: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    technology: List[str] = field(default_factory=list)
    supports_page_source: bool = False
    supports_application_state: bool = False
    requires_type_conversion: bool = False
    supports_async: bool = False


@dataclass(frozen=True)
class LibraryMetadata:
    """Primary metadata describing a Robot Framework library plugin."""

    name: str
    package_name: str
    import_path: str
    description: str
    library_type: Literal["builtin", "external"]
    use_cases: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    installation_command: str = ""
    post_install_commands: List[str] = field(default_factory=list)
    platform_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    supports_async: bool = False
    is_deprecated: bool = False
    requires_type_conversion: bool = False
    load_priority: int = 100
    default_enabled: bool = True
    extra_name: Optional[str] = None
    schema_version: int = 1
    technology_tags: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


class LibraryStateProvider(Protocol):
    """State provider interface for page source or application state retrieval."""

    async def get_page_source(
        self,
        session: "ExecutionSession",
        *,
        full_source: bool = False,
        filtered: bool = False,
        filtering_level: str = "standard",
        include_reduced_dom: bool = True,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        ...

    async def get_application_state(
        self,
        session: "ExecutionSession",
    ) -> Optional[Dict[str, Any]]:
        ...


class TypeConversionProvider(Protocol):
    """Interface for plugins that provide custom type conversion."""

    def requires_type_conversion(self) -> bool:
        """Return True when the library requires Robot Framework type conversion."""

    def get_conversion_map(self) -> Dict[str, Any]:
        """Return optional conversion metadata for advanced scenarios."""


KeywordOverrideHandler = Callable[
    [
        "ExecutionSession",
        str,
        List[str],
        Optional[Any],
    ],
    Awaitable[Optional[Dict[str, Any]]],
]


@runtime_checkable
class LibraryPlugin(Protocol):
    """Protocol describing a library plugin implementation."""

    schema_version: int = 1

    def get_metadata(self) -> LibraryMetadata:
        """Return metadata for the plugin."""

    def get_capabilities(self) -> Optional[LibraryCapabilities]:
        """Return capability information for the plugin."""

    def get_install_actions(self) -> Optional[List[InstallAction]]:
        """Return installation actions."""

    def get_hints(self) -> Optional[LibraryHints]:
        """Return optional hints and examples."""

    def get_prompt_bundle(self) -> Optional[PromptBundle]:
        """Return optional prompt bundle."""

    def get_state_provider(self) -> Optional[LibraryStateProvider]:
        """Return state provider if available."""

    def get_type_converters(self) -> Optional[TypeConversionProvider]:
        """Return type conversion provider if available."""

    def get_keyword_library_map(self) -> Optional[Dict[str, str]]:
        """Return mapping of keyword names to this library."""

    def get_keyword_overrides(self) -> Optional[Dict[str, KeywordOverrideHandler]]:
        """Return async override handlers for specific keywords."""

    def before_keyword_execution(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        library_manager: Any,
        keyword_discovery: Any,
    ) -> None:
        """Hook executed before a keyword is run."""

    def on_session_start(self, session: "ExecutionSession") -> None:
        """Hook executed when a new session starts."""

    def on_session_end(self, session: "ExecutionSession") -> None:
        """Hook executed when a session ends."""


# Import guarded to avoid circular dependency during runtime import
try:  # pragma: no cover - imported lazily for typing only
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover - module not available at import time
    ExecutionSession = Any  # type: ignore
