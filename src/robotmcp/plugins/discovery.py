"""Plugin discovery utilities for rf-mcp."""

from __future__ import annotations

import json
import logging
import os
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterator, List, Optional

from .base import ManifestLibraryPlugin, StaticLibraryPlugin
from .contracts import LibraryCapabilities, LibraryMetadata

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT = "robotmcp.library_plugins"
DEFAULT_MANIFEST_DIR = ".robotmcp/plugins"


def iter_entry_point_plugins() -> Iterator[StaticLibraryPlugin]:
    """Yield plugins discovered via Python entry points."""
    try:
        entry_points = importlib_metadata.entry_points()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load entry points for plugins: %s", exc)
        return iter(())

    group = getattr(entry_points, "select", None)
    if group:
        candidates = group(group=PLUGIN_ENTRY_POINT)
    else:  # Python <3.10 compatibility
        candidates = [
            ep for ep in entry_points.get(PLUGIN_ENTRY_POINT, [])  # type: ignore[attr-defined]
        ]

    for ep in candidates:
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            yield plugin
        except Exception as exc:  # pragma: no cover - plugin errors
            logger.warning("Failed to load plugin from entry point %s: %s", ep, exc)


def iter_manifest_plugins(paths: Optional[List[str]] = None) -> Iterator[StaticLibraryPlugin]:
    """Yield plugins defined via JSON manifests."""
    manifest_paths = _expand_manifest_paths(paths)
    for manifest_path in manifest_paths:
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Failed to read plugin manifest %s: %s", manifest_path, exc)
            continue

        plugin = _build_plugin_from_manifest(data, manifest_path)
        if plugin:
            yield plugin


def _expand_manifest_paths(paths: Optional[List[str]]) -> List[Path]:
    env_paths = os.environ.get("ROBOTMCP_PLUGIN_PATHS")
    resolved_paths: List[str] = []

    if paths:
        resolved_paths.extend(paths)
    if env_paths:
        resolved_paths.extend(path.strip() for path in env_paths.split(os.pathsep) if path.strip())
    if not resolved_paths:
        resolved_paths.append(DEFAULT_MANIFEST_DIR)

    manifest_files: List[Path] = []
    for path_str in resolved_paths:
        path = Path(path_str).expanduser()
        if path.is_dir():
            manifest_files.extend(sorted(path.glob("*.json")))
        elif path.suffix == ".json" and path.exists():
            manifest_files.append(path)
    return manifest_files


def _build_plugin_from_manifest(data: dict, source: Path) -> Optional[StaticLibraryPlugin]:
    module_path = data.get("module")
    plugin_class = data.get("class")
    metadata_dict = data.get("metadata")

    try:
        if module_path and plugin_class:
            module = import_module(module_path)
            plugin_cls = getattr(module, plugin_class)
            init_kwargs = data.get("init_kwargs") or {}
            return plugin_cls(**init_kwargs)

        if metadata_dict:
            metadata = LibraryMetadata(
                name=metadata_dict["name"],
                package_name=metadata_dict["package_name"],
                import_path=metadata_dict.get("import_path", metadata_dict["name"]),
                description=metadata_dict.get("description", metadata_dict["name"]),
                library_type=metadata_dict.get("library_type", "external"),
                use_cases=metadata_dict.get("use_cases", []),
                categories=metadata_dict.get("categories", []),
                contexts=metadata_dict.get("contexts", []),
                installation_command=metadata_dict.get("installation_command", ""),
                post_install_commands=metadata_dict.get("post_install_commands", []),
                platform_requirements=metadata_dict.get("platform_requirements", []),
                dependencies=metadata_dict.get("dependencies", []),
                supports_async=metadata_dict.get("supports_async", False),
                is_deprecated=metadata_dict.get("is_deprecated", False),
                requires_type_conversion=metadata_dict.get(
                    "requires_type_conversion", False
                ),
                load_priority=metadata_dict.get("load_priority", 100),
                default_enabled=metadata_dict.get("default_enabled", True),
                extra_name=metadata_dict.get("extra_name"),
                schema_version=metadata_dict.get("schema_version", 1),
                technology_tags=metadata_dict.get("technology_tags", []),
                tags=metadata_dict.get("tags", []),
                aliases=metadata_dict.get("aliases", []),
            )
            capabilities_dict = metadata_dict.get("capabilities", {})
            capabilities = LibraryCapabilities(
                contexts=capabilities_dict.get("contexts", metadata.contexts),
                features=capabilities_dict.get("features", []),
                technology=capabilities_dict.get("technology", metadata.technology_tags),
                supports_page_source=capabilities_dict.get("supports_page_source", False),
                supports_application_state=capabilities_dict.get(
                    "supports_application_state", False
                ),
                requires_type_conversion=metadata.requires_type_conversion,
                supports_async=capabilities_dict.get(
                    "supports_async", metadata.supports_async
                ),
            )
            return ManifestLibraryPlugin(metadata=metadata, capabilities=capabilities)

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to build plugin from manifest %s: %s", source, exc)

    logger.warning("Invalid plugin manifest at %s; skipping", source)
    return None

