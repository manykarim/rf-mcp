"""Helpers for builtin rf-mcp library plugins."""

from __future__ import annotations

from typing import List, Sequence

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import InstallAction, LibraryCapabilities, LibraryMetadata

from .definitions import BUILTIN_LIBRARY_DEFINITIONS


def _infer_contexts(categories: Sequence[str]) -> List[str]:
    inferred = []
    for category in categories:
        if category in {"web", "mobile", "api", "desktop"}:
            inferred.append(category)
    seen = set()
    ordered: List[str] = []
    for ctx in inferred:
        if ctx not in seen:
            ordered.append(ctx)
            seen.add(ctx)
    return ordered


def generate_builtin_plugins() -> List[StaticLibraryPlugin]:
    """Generate StaticLibraryPlugin instances for builtin definitions."""

    plugins: List[StaticLibraryPlugin] = []
    for definition in BUILTIN_LIBRARY_DEFINITIONS:
        categories = definition.get("categories", [])
        contexts = definition.get("contexts") or _infer_contexts(categories)

        metadata = LibraryMetadata(
            name=definition["name"],
            package_name=definition["package_name"],
            import_path=definition["import_path"],
            description=definition["description"],
            library_type=definition["library_type"],
            use_cases=definition.get("use_cases", []),
            categories=categories,
            contexts=contexts,
            installation_command=definition.get("installation_command", ""),
            post_install_commands=definition.get("post_install_commands", []),
            platform_requirements=definition.get("platform_requirements", []),
            dependencies=definition.get("dependencies", []),
            supports_async=definition.get("supports_async", False),
            is_deprecated=definition.get("is_deprecated", False),
            requires_type_conversion=definition.get("requires_type_conversion", False),
            load_priority=definition.get("load_priority", 100),
            default_enabled=definition.get("default_enabled", True),
            extra_name=definition.get("extra_name"),
        )

        capabilities = LibraryCapabilities(
            contexts=contexts,
            features=[],
            technology=definition.get("technology", []),
            supports_page_source=definition.get("supports_page_source", False),
            supports_application_state=definition.get("supports_application_state", False),
            requires_type_conversion=metadata.requires_type_conversion,
            supports_async=metadata.supports_async,
        )

        install_actions = []
        install_cmd = metadata.installation_command.strip()
        if install_cmd and "built-in" not in install_cmd.lower():
            install_actions.append(
                InstallAction(
                    description=f"Install {metadata.name}",
                    command=install_cmd.split(),
                )
            )
        for cmd in metadata.post_install_commands:
            install_actions.append(
                InstallAction(
                    description=f"Post-install for {metadata.name}",
                    command=cmd.split(),
                )
            )

        plugin = StaticLibraryPlugin(
            metadata=metadata,
            capabilities=capabilities,
            install_actions=install_actions or None,
        )
        plugins.append(plugin)

    return plugins


__all__ = ["generate_builtin_plugins"]

