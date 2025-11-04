"""Centralized Robot Framework Library Registry.

This module provides a single source of truth for all Robot Framework libraries
supported by the MCP server. It eliminates duplication across components and
ensures consistent library support.
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from robotmcp.plugins import get_library_plugin_manager
from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryMetadata,
)



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
    # Basic identification
    name: str
    package_name: str
    import_path: str
    library_type: LibraryType
    
    # Descriptive information
    description: str
    use_cases: List[str] = field(default_factory=list)
    categories: List[LibraryCategory] = field(default_factory=list)
    
    # Installation and setup
    installation_command: str = ""
    post_install_commands: List[str] = field(default_factory=list)
    platform_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Feature flags
    requires_type_conversion: bool = False
    supports_async: bool = False
    is_deprecated: bool = False
    
    # Priority and filtering
    load_priority: int = 100  # Lower numbers = higher priority
    default_enabled: bool = True
    extra_name: Optional[str] = None
    
    @property
    def is_builtin(self) -> bool:
        """Check if this is a built-in Robot Framework library."""
        return self.library_type == LibraryType.BUILTIN
    
    @property
    def is_external(self) -> bool:
        """Check if this is an external library requiring installation."""
        return self.library_type == LibraryType.EXTERNAL
    
    def has_category(self, category: LibraryCategory) -> bool:
        """Check if library belongs to a specific category."""
        return category in self.categories


# ============================================================================
# ROBOT FRAMEWORK LIBRARY REGISTRY
# ============================================================================

ROBOT_FRAMEWORK_LIBRARIES: Dict[str, LibraryConfig] = {
    
    # ========================================================================
    # BUILT-IN ROBOT FRAMEWORK LIBRARIES
    # ========================================================================
    
    'BuiltIn': LibraryConfig(
        name='BuiltIn',
        package_name='robotframework',
        import_path='robot.libraries.BuiltIn',
        library_type=LibraryType.BUILTIN,
        description='Robot Framework built-in library with generic keywords',
        use_cases=['basic operations', 'variables', 'control flow', 'logging', 'keywords'],
        categories=[LibraryCategory.CORE, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=1  # Highest priority
    ),
    
    'Collections': LibraryConfig(
        name='Collections',
        package_name='robotframework',
        import_path='robot.libraries.Collections',
        library_type=LibraryType.BUILTIN,
        description='Keywords for handling lists and dictionaries',
        use_cases=['list manipulation', 'dictionary operations', 'data structures'],
        categories=[LibraryCategory.CORE, LibraryCategory.DATA, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=10
    ),
    
    'DateTime': LibraryConfig(
        name='DateTime',
        package_name='robotframework',
        import_path='robot.libraries.DateTime',
        library_type=LibraryType.BUILTIN,
        description='Date and time manipulation keywords',
        use_cases=['date operations', 'time calculations', 'timestamp validation'],
        categories=[LibraryCategory.CORE, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=20
    ),
    
    'Dialogs': LibraryConfig(
        name='Dialogs',
        package_name='robotframework',
        import_path='robot.libraries.Dialogs',
        library_type=LibraryType.BUILTIN,
        description='User interaction and pause execution keywords',
        use_cases=['user input', 'pause execution', 'interactive testing', 'manual intervention'],
        categories=[LibraryCategory.CORE, LibraryCategory.TESTING, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=50
    ),
    
    'OperatingSystem': LibraryConfig(
        name='OperatingSystem',
        package_name='robotframework',
        import_path='robot.libraries.OperatingSystem',
        library_type=LibraryType.BUILTIN,
        description='Operating system related keywords',
        use_cases=['file operations', 'directory management', 'environment variables', 'system commands'],
        categories=[LibraryCategory.CORE, LibraryCategory.SYSTEM, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=15
    ),
    
    'Process': LibraryConfig(
        name='Process',
        package_name='robotframework',
        import_path='robot.libraries.Process',
        library_type=LibraryType.BUILTIN,
        description='Process execution and management keywords',
        use_cases=['execute commands', 'process management', 'system integration'],
        categories=[LibraryCategory.CORE, LibraryCategory.SYSTEM, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=25
    ),
    
    'Screenshot': LibraryConfig(
        name='Screenshot',
        package_name='robotframework',
        import_path='robot.libraries.Screenshot',
        library_type=LibraryType.BUILTIN,
        description='Desktop screenshot capture keywords',
        use_cases=['desktop screenshots', 'visual documentation', 'debugging'],
        categories=[LibraryCategory.CORE, LibraryCategory.VISUAL, LibraryCategory.TESTING],
        installation_command='Built-in with Robot Framework',
        load_priority=40
    ),
    
    'String': LibraryConfig(
        name='String',
        package_name='robotframework',
        import_path='robot.libraries.String',
        library_type=LibraryType.BUILTIN,
        description='String manipulation and validation keywords',
        use_cases=['text manipulation', 'string validation', 'pattern matching'],
        categories=[LibraryCategory.CORE, LibraryCategory.DATA, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        load_priority=12
    ),
    
    'Telnet': LibraryConfig(
        name='Telnet',
        package_name='robotframework',
        import_path='robot.libraries.Telnet',
        library_type=LibraryType.BUILTIN,
        description='Telnet connection and command execution keywords',
        use_cases=['telnet connections', 'network protocols', 'legacy systems'],
        categories=[LibraryCategory.CORE, LibraryCategory.NETWORK, LibraryCategory.SYSTEM],
        installation_command='Built-in with Robot Framework',
        load_priority=45
    ),
    
    'XML': LibraryConfig(
        name='XML',
        package_name='robotframework',
        import_path='robot.libraries.XML',
        library_type=LibraryType.BUILTIN,
        description='XML parsing, validation and manipulation keywords',
        use_cases=['xml parsing', 'xml validation', 'data verification'],
        categories=[LibraryCategory.CORE, LibraryCategory.DATA, LibraryCategory.UTILITIES],
        installation_command='Built-in with Robot Framework',
        requires_type_conversion=False,  # XML uses basic types
        load_priority=30
    ),
    
    # ========================================================================
    # EXTERNAL ROBOT FRAMEWORK LIBRARIES
    # ========================================================================
    
    'Browser': LibraryConfig(
        name='Browser',
        package_name='robotframework-browser',
        import_path='Browser',
        library_type=LibraryType.EXTERNAL,
        extra_name='web',
        description='Modern web testing with Playwright backend',
        use_cases=['modern web testing', 'playwright automation', 'web performance', 'mobile web'],
        categories=[LibraryCategory.WEB, LibraryCategory.TESTING],
        installation_command='pip install robotframework-browser',
        post_install_commands=['rfbrowser init'],
        dependencies=['playwright', 'node.js'],
        requires_type_conversion=True,  # Browser uses complex enums
        supports_async=True,
        load_priority=5  # High priority for modern web testing
    ),
    
    'SeleniumLibrary': LibraryConfig(
        name='SeleniumLibrary',
        package_name='robotframework-seleniumlibrary',
        import_path='SeleniumLibrary',
        library_type=LibraryType.EXTERNAL,
        extra_name='web',
        description='Traditional web testing with Selenium WebDriver',
        use_cases=['web testing', 'browser automation', 'web elements', 'form filling'],
        categories=[LibraryCategory.WEB, LibraryCategory.TESTING],
        installation_command='pip install robotframework-seleniumlibrary',
        dependencies=['selenium'],
        requires_type_conversion=True,  # Selenium has Union types, bools, etc.
        load_priority=8  # Lower priority than Browser Library
    ),
    
    'RequestsLibrary': LibraryConfig(
        name='RequestsLibrary',
        package_name='robotframework-requests',
        import_path='RequestsLibrary',
        library_type=LibraryType.EXTERNAL,
        extra_name='api',
        description='HTTP API testing by wrapping Python Requests Library',
        use_cases=['api testing', 'http requests', 'rest api', 'json validation'],
        categories=[LibraryCategory.API, LibraryCategory.TESTING, LibraryCategory.NETWORK],
        installation_command='pip install robotframework-requests',
        requires_type_conversion=True,  # Requests has timeout, dict parameters
        load_priority=6
    ),
    
    'DatabaseLibrary': LibraryConfig(
        name='DatabaseLibrary',
        package_name='robotframework-databaselibrary',
        import_path='DatabaseLibrary',
        library_type=LibraryType.EXTERNAL,
        extra_name='database',
        description='Database testing with multiple DB support',
        use_cases=['database testing', 'sql queries', 'data validation', 'database connections'],
        categories=[LibraryCategory.DATABASE, LibraryCategory.TESTING, LibraryCategory.DATA],
        installation_command='pip install robotframework-databaselibrary',
        load_priority=22
    ),
    
    'AppiumLibrary': LibraryConfig(
        name='AppiumLibrary',
        package_name='robotframework-appiumlibrary',
        import_path='AppiumLibrary',
        library_type=LibraryType.EXTERNAL,
        extra_name='mobile',
        description='Mobile app testing with Appium',
        use_cases=['mobile testing', 'android automation', 'ios testing', 'app testing'],
        categories=[LibraryCategory.MOBILE, LibraryCategory.TESTING],
        installation_command='pip install robotframework-appiumlibrary',
        dependencies=['appium'],
        load_priority=27,
        default_enabled=False  # PHASE 3: Disable by default to prevent conflicts with web testing
    ),
    
    'SSHLibrary': LibraryConfig(
        name='SSHLibrary',
        package_name='robotframework-sshlibrary',
        import_path='SSHLibrary',
        library_type=LibraryType.EXTERNAL,
        description='SSH and SFTP operations',
        use_cases=['remote connections', 'ssh commands', 'file transfer', 'server management'],
        categories=[LibraryCategory.NETWORK, LibraryCategory.SYSTEM],
        installation_command='pip install robotframework-sshlibrary',
        load_priority=31
    ),
    
    'FTPLibrary': LibraryConfig(
        name='FTPLibrary',
        package_name='robotframework-ftplibrary',
        import_path='FTPLibrary',
        library_type=LibraryType.EXTERNAL,
        description='FTP operations and file transfer',
        use_cases=['ftp operations', 'file transfer', 'server management'],
        categories=[LibraryCategory.NETWORK, LibraryCategory.SYSTEM],
        installation_command='pip install robotframework-ftplibrary',
        load_priority=35
    ),
    
}

logger = logging.getLogger(__name__)
_PLUGIN_STATE_LOCK = threading.Lock()
_PLUGINS_REGISTERED = False


def _config_to_metadata(config: LibraryConfig) -> LibraryMetadata:
    return LibraryMetadata(
        name=config.name,
        package_name=config.package_name,
        import_path=config.import_path,
        description=config.description,
        library_type=config.library_type.value,
        use_cases=list(config.use_cases),
        categories=[category.value for category in config.categories],
        contexts=[],
        installation_command=config.installation_command,
        post_install_commands=list(config.post_install_commands),
        platform_requirements=list(config.platform_requirements),
        dependencies=list(config.dependencies),
        supports_async=config.supports_async,
        is_deprecated=config.is_deprecated,
        requires_type_conversion=config.requires_type_conversion,
        load_priority=config.load_priority,
        default_enabled=config.default_enabled,
        extra_name=config.extra_name,
    )


def _config_to_capabilities(config: LibraryConfig) -> LibraryCapabilities:
    contexts = [
        category.value
        for category in config.categories
        if category.value in {"web", "mobile", "api", "desktop"}
    ]
    return LibraryCapabilities(
        contexts=contexts,
        features=[],
        technology=[],
        supports_page_source=False,
        supports_application_state=False,
        requires_type_conversion=config.requires_type_conversion,
        supports_async=config.supports_async,
    )


def _config_to_install_actions(config: LibraryConfig) -> List[InstallAction]:
    actions: List[InstallAction] = []
    command = config.installation_command.strip()
    if command and "built-in" not in command.lower():
        actions.append(
            InstallAction(
                description=f"Install {config.name}",
                command=[command],
            )
        )
    for idx, post_command in enumerate(config.post_install_commands, start=1):
        if post_command:
            actions.append(
                InstallAction(
                    description=f"Post-install step {idx} for {config.name}",
                    command=[post_command],
                )
            )
    return actions


def _metadata_to_config(
    metadata: LibraryMetadata,
    capabilities: Optional[LibraryCapabilities],
) -> LibraryConfig:
    categories: List[LibraryCategory] = []
    for category in metadata.categories:
        try:
            categories.append(LibraryCategory(category))
        except ValueError:
            logger.debug("Unknown library category '%s' for %s; skipping", category, metadata.name)

    library_type = (
        LibraryType.BUILTIN
        if metadata.library_type == LibraryType.BUILTIN.value
        else LibraryType.EXTERNAL
    )

    requires_type_conversion = metadata.requires_type_conversion or (
        capabilities.requires_type_conversion if capabilities else False
    )
    supports_async = metadata.supports_async or (
        capabilities.supports_async if capabilities else False
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
        requires_type_conversion=requires_type_conversion,
        supports_async=supports_async,
        is_deprecated=metadata.is_deprecated,
        load_priority=metadata.load_priority,
        default_enabled=metadata.default_enabled,
        extra_name=metadata.extra_name,
    )


def _ensure_plugins_registered() -> None:
    global _PLUGINS_REGISTERED
    if _PLUGINS_REGISTERED:
        return
    with _PLUGIN_STATE_LOCK:
        if _PLUGINS_REGISTERED:
            return
        manager = get_library_plugin_manager()
        builtin_plugins: List[StaticLibraryPlugin] = []
        for config in ROBOT_FRAMEWORK_LIBRARIES.values():
            metadata = _config_to_metadata(config)
            capabilities = _config_to_capabilities(config)
            install_actions = _config_to_install_actions(config)
            plugin = StaticLibraryPlugin(
                metadata=metadata,
                capabilities=capabilities,
                install_actions=install_actions or None,
            )
            builtin_plugins.append(plugin)
        manager.register_plugins(builtin_plugins, source="builtin")
        # Discover external plugins through entry points and manifests
        manager.discover_entry_point_plugins()
        manager.discover_manifest_plugins()
        _PLUGINS_REGISTERED = True


def _reset_plugin_state_for_tests() -> None:
    """Reset plugin registration state (testing helper)."""
    global _PLUGINS_REGISTERED
    with _PLUGIN_STATE_LOCK:
        _PLUGINS_REGISTERED = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _build_config_snapshot() -> Dict[str, LibraryConfig]:
    _ensure_plugins_registered()
    manager = get_library_plugin_manager()
    configs: Dict[str, LibraryConfig] = {
        name: _metadata_to_config(meta, manager.get_capabilities(name))
        for name, meta in manager.iter_metadata()
    }
    # Ensure legacy entries remain accessible if plugin discovery failed
    for name, legacy in ROBOT_FRAMEWORK_LIBRARIES.items():
        configs.setdefault(name, legacy)
    return configs


def get_all_libraries() -> Dict[str, LibraryConfig]:
    """Get all registered Robot Framework libraries."""
    return _build_config_snapshot()


def get_library_config(library_name: str) -> Optional[LibraryConfig]:
    """Fetch a single library configuration by name."""
    _ensure_plugins_registered()
    manager = get_library_plugin_manager()
    metadata = manager.get_metadata(library_name)
    if metadata:
        capabilities = manager.get_capabilities(library_name)
        return _metadata_to_config(metadata, capabilities)
    return ROBOT_FRAMEWORK_LIBRARIES.get(library_name)


def get_library_extra_name(library_name: str) -> Optional[str]:
    """Return the optional dependency extra tied to a library, if any."""
    config = get_library_config(library_name)
    return config.extra_name if config else None


def get_library_install_hint(library_name: str) -> Optional[str]:
    """Return a human-friendly installation hint for a library."""
    config = get_library_config(library_name)
    if not config:
        return None

    extra_name = config.extra_name
    if extra_name:
        return (
            f"Install via `pip install rf-mcp[{extra_name}]` or `{config.installation_command}`."
            if config.installation_command
            else f"Install via `pip install rf-mcp[{extra_name}]`."
        )

    if config.installation_command and "built-in" not in config.installation_command.lower():
        return f"Install via `{config.installation_command}`."

    return None


def get_builtin_libraries() -> Dict[str, LibraryConfig]:
    """Get only built-in Robot Framework libraries."""
    return {
        name: lib
        for name, lib in _build_config_snapshot().items()
        if lib.is_builtin
    }


def get_external_libraries() -> Dict[str, LibraryConfig]:
    """Get only external Robot Framework libraries."""
    return {
        name: lib
        for name, lib in _build_config_snapshot().items()
        if lib.is_external
    }


def get_libraries_by_category(category: LibraryCategory) -> Dict[str, LibraryConfig]:
    """Get libraries belonging to a specific category."""
    return {
        name: lib
        for name, lib in _build_config_snapshot().items()
        if lib.has_category(category)
    }


def get_libraries_requiring_type_conversion() -> List[str]:
    """Get list of library names that require Robot Framework type conversion."""
    return [
        lib.name
        for lib in _build_config_snapshot().values()
        if lib.requires_type_conversion
    ]


def get_library_names_for_loading() -> List[str]:
    """Get ordered list of library names for loading (by priority)."""
    libs = sorted(
        _build_config_snapshot().values(),
        key=lambda x: x.load_priority,
    )
    return [lib.name for lib in libs if lib.default_enabled]


def get_installation_info() -> Dict[str, Dict[str, Any]]:
    """Get installation information for all libraries (for library_checker compatibility)."""
    libraries = _build_config_snapshot()
    return {
        name: {
            "package": lib.package_name,
            "import": lib.import_path,
            "description": lib.description,
            "is_builtin": lib.is_builtin,
            "post_install": lib.post_install_commands[0]
            if lib.post_install_commands
            else None,
        }
        for name, lib in libraries.items()
    }


def get_recommendation_info() -> List[Dict[str, Any]]:
    """Get library information for recommendations (for library_recommender compatibility)."""
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


# ============================================================================
# VALIDATION
# ============================================================================


def validate_registry() -> List[str]:
    """Validate the library registry for consistency and completeness."""
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


# Run validation on import
_VALIDATION_ERRORS = validate_registry()
if _VALIDATION_ERRORS:
    import warnings

    warnings.warn(f"Library registry validation errors: {_VALIDATION_ERRORS}")
