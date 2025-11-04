"""Centralized Robot Framework Library Registry backed by the plugin system."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from robotmcp.plugins import (
    get_library_plugin_manager,
    reset_library_plugin_manager_for_tests,
)
from robotmcp.plugins.builtin import generate_builtin_plugins
from robotmcp.plugins.contracts import LibraryMetadata

logger = logging.getLogger(__name__)


class LibraryType(Enum):
    """Type of Robot Framework library."""

    BUILTIN = "builtin"  # Built into Robot Framework
    EXTERNAL = "external"  # Third-party libraries requiring installation


class LibraryCategory(Enum):
    """Categories for library organization and recommendations."""

    CORE = "core"
    WEB = "web"
    API = "api"
    MOBILE = "mobile"
    DATABASE = "database"
    DATA = "data"
    SYSTEM = "system"
    NETWORK = "network"
    VISUAL = "visual"
    TESTING = "testing"
    UTILITIES = "utilities"


@dataclass
class LibraryConfig:
    """Configuration for a Robot Framework library."""

    name: str
    package_name: str
    import_path: str
    library_type: LibraryType

    description: str
    use_cases: List[str] = field(default_factory=list)
    categories: List[LibraryCategory] = field(default_factory=list)

    installation_command: str = ""
    post_install_commands: List[str] = field(default_factory=list)
    platform_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    requires_type_conversion: bool = False
    supports_async: bool = False
    is_deprecated: bool = False

    load_priority: int = 100
    default_enabled: bool = True
    extra_name: Optional[str] = None

    @property
    def is_builtin(self) -> bool:
        return self.library_type == LibraryType.BUILTIN

    @property
    def is_external(self) -> bool:
        return self.library_type == LibraryType.EXTERNAL

    def has_category(self, category: LibraryCategory) -> bool:
        return category in self.categories


_PLUGIN_STATE_LOCK = threading.Lock()
_PLUGINS_REGISTERED = False


def _metadata_to_config(metadata: LibraryMetadata) -> LibraryConfig:
    categories: List[LibraryCategory] = []
    for value in metadata.categories:
        try:
            categories.append(LibraryCategory(value))
        except ValueError:
            logger.debug(
                "Unknown library category '%s' for %s; ignoring", value, metadata.name
            )

    library_type = (
        LibraryType.BUILTIN
        if metadata.library_type == LibraryType.BUILTIN.value
        else LibraryType.EXTERNAL
    )

    return LibraryConfig(
        name=metadata.name,
        package_name=metadata.package_name,
        import_path=metadata.import_path,
        library_type=library_type,
        description=metadata.description,
        use_cases=list(metadata.use_cases),
        categories=categories,
        installation_command=metadata.installation_command,
        post_install_commands=list(metadata.post_install_commands),
        platform_requirements=list(metadata.platform_requirements),
        dependencies=list(metadata.dependencies),
        requires_type_conversion=metadata.requires_type_conversion,
        supports_async=metadata.supports_async,
        is_deprecated=metadata.is_deprecated,
        load_priority=metadata.load_priority,
        default_enabled=metadata.default_enabled,
        extra_name=metadata.extra_name,
    )


def _initialize_library_plugins(paths: Optional[List[str]] = None) -> None:
    manager = get_library_plugin_manager()
    manager.register_plugins(generate_builtin_plugins(), source="builtin")
    manager.discover_entry_point_plugins()
    manager.discover_manifest_plugins(paths)


def _ensure_plugins_registered(paths: Optional[List[str]] = None) -> None:
    global _PLUGINS_REGISTERED
    if _PLUGINS_REGISTERED:
        return
    with _PLUGIN_STATE_LOCK:
        if _PLUGINS_REGISTERED:
            return
        _initialize_library_plugins(paths)
        _PLUGINS_REGISTERED = True


def _reset_plugin_state_for_tests() -> None:
    global _PLUGINS_REGISTERED
    with _PLUGIN_STATE_LOCK:
        _PLUGINS_REGISTERED = False
    reset_library_plugin_manager_for_tests()


def reload_library_plugins(paths: Optional[List[str]] = None) -> Dict[str, LibraryConfig]:
    """Force plugin reload and return the resulting library snapshot."""
    _reset_plugin_state_for_tests()
    _ensure_plugins_registered(paths)
    return _build_config_snapshot()


def _build_config_snapshot() -> Dict[str, LibraryConfig]:
    _ensure_plugins_registered()
    manager = get_library_plugin_manager()
    configs: Dict[str, LibraryConfig] = {}
    for name in manager.list_plugin_names():
        metadata = manager.get_metadata(name)
        if not metadata:
            continue
        configs[name] = _metadata_to_config(metadata)
    return configs


# ============================================================================
# PUBLIC API
# ============================================================================


def get_all_libraries() -> Dict[str, LibraryConfig]:
    """Get all registered Robot Framework libraries."""
    return _build_config_snapshot()


def get_library_config(library_name: str) -> Optional[LibraryConfig]:
    """Fetch a single library configuration by name."""
    configs = _build_config_snapshot()
    return configs.get(library_name)


def get_library_extra_name(library_name: str) -> Optional[str]:
    config = get_library_config(library_name)
    return config.extra_name if config else None


def get_library_install_hint(library_name: str) -> Optional[str]:
    config = get_library_config(library_name)
    if not config:
        return None

    extra_name = config.extra_name
    if extra_name:
        if config.installation_command:
            return (
                f"Install via `pip install rf-mcp[{extra_name}]` "
                f"or `{config.installation_command}`."
            )
        return f"Install via `pip install rf-mcp[{extra_name}]`."

    if config.installation_command and "built-in" not in config.installation_command.lower():
        return f"Install via `{config.installation_command}`."

    return None


def get_builtin_libraries() -> Dict[str, LibraryConfig]:
    return {
        name: config
        for name, config in _build_config_snapshot().items()
        if config.is_builtin
    }


def get_external_libraries() -> Dict[str, LibraryConfig]:
    return {
        name: config
        for name, config in _build_config_snapshot().items()
        if config.is_external
    }


def get_libraries_by_category(category: LibraryCategory) -> Dict[str, LibraryConfig]:
    return {
        name: config
        for name, config in _build_config_snapshot().items()
        if config.has_category(category)
    }


def get_libraries_requiring_type_conversion() -> List[str]:
    return [
        config.name
        for config in _build_config_snapshot().values()
        if config.requires_type_conversion
    ]


def get_library_names_for_loading() -> List[str]:
    configs = sorted(
        _build_config_snapshot().values(),
        key=lambda cfg: cfg.load_priority,
    )
    return [cfg.name for cfg in configs if cfg.default_enabled]


def get_installation_info() -> Dict[str, Dict[str, Any]]:
    libraries = _build_config_snapshot()
    return {
        name: {
            "package": lib.package_name,
            "import": lib.import_path,
            "description": lib.description,
            "is_builtin": lib.is_builtin,
            "post_install": lib.post_install_commands[0] if lib.post_install_commands else None,
        }
        for name, lib in libraries.items()
    }


def get_recommendation_info() -> List[Dict[str, Any]]:
    libraries = _build_config_snapshot()
    return [
        {
            "name": lib.name,
            "package_name": lib.package_name,
            "installation_command": lib.installation_command,
            "use_cases": lib.use_cases,
            "categories": [cat.value for cat in lib.categories],
            "description": lib.description,
            "is_builtin": lib.is_builtin,
            "requires_setup": bool(lib.post_install_commands),
            "setup_commands": lib.post_install_commands,
            "platform_requirements": lib.platform_requirements,
            "dependencies": lib.dependencies,
        }
        for lib in libraries.values()
    ]


def validate_registry() -> List[str]:
    errors: List[str] = []
    libraries = _build_config_snapshot()

    for name, lib in libraries.items():
        if not lib.name:
            errors.append(f"Library {name}: Missing name")
        if not lib.package_name:
            errors.append(f"Library {name}: Missing package_name")
        if not lib.import_path:
            errors.append(f"Library {name}: Missing import_path")
        if not lib.description:
            errors.append(f"Library {name}: Missing description")

        if lib.is_builtin and lib.package_name != "robotframework":
            errors.append(
                f"Library {name}: Built-in library should have package_name='robotframework'"
            )

        if lib.is_external and not lib.installation_command:
            errors.append(
                f"Library {name}: External library missing installation_command"
            )

    priorities: Dict[int, str] = {}
    for name, lib in libraries.items():
        if lib.load_priority in priorities:
            errors.append(
                f"Duplicate priority {lib.load_priority}: {name} and {priorities[lib.load_priority]}"
            )
        priorities[lib.load_priority] = name

    return errors


_VALIDATION_ERRORS = validate_registry()
if _VALIDATION_ERRORS:
    import warnings

    warnings.warn(f"Library registry validation errors: {_VALIDATION_ERRORS}")


__all__ = [
    "LibraryCategory",
    "LibraryConfig",
    "LibraryType",
    "get_all_libraries",
    "get_library_config",
    "get_library_extra_name",
    "get_library_install_hint",
    "get_builtin_libraries",
    "get_external_libraries",
    "get_libraries_by_category",
    "get_libraries_requiring_type_conversion",
    "get_library_names_for_loading",
    "get_installation_info",
    "get_recommendation_info",
    "reload_library_plugins",
]

