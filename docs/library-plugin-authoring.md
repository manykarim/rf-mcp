# Library Plugin Authoring Guide

This guide explains how to build and distribute plugins that extend `rf-mcp` with new Robot Framework libraries (metadata, hooks, page-source/state providers, type conversion, prompts).

## Plugin Overview

A plugin supplies:

- **Metadata**: name, import path, description, categories, etc.
- **Optional hooks**: lifecycle (`on_session_start`, `on_session_end`), install actions, hints, prompt snippets.
- **Capability providers**: page-source/application state retrieval, type conversion helpers.
- **Discovery information**: either a Python entry point (`robotmcp.library_plugins`) or a local manifest (`.json` file).

The plugin interface lives in `src/robotmcp/plugins/contracts.py`:

```python
class LibraryPlugin(Protocol):
    schema_version: int = 1

    def get_metadata(self) -> LibraryMetadata: ...
    def get_capabilities(self) -> Optional[LibraryCapabilities]: ...
    def get_install_actions(self) -> Optional[List[InstallAction]]: ...
    def get_hints(self) -> Optional[LibraryHints]: ...
    def get_prompt_bundle(self) -> Optional[PromptBundle]: ...
    def get_state_provider(self) -> Optional[LibraryStateProvider]: ...
    def get_type_converters(self) -> Optional[TypeConversionProvider]: ...
    def on_session_start(self, session: ExecutionSession) -> None: ...
    def on_session_end(self, session: ExecutionSession) -> None: ...
```

Only `get_metadata` is required; every other method is optional.

## Quick Start (Python Package)

1. Add `rf-mcp` as a dependency (or assume the target env already has it).
2. Create a module implementing `LibraryPlugin`.

```python
# my_plugins/browser_plus.py
from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import LibraryCapabilities, LibraryMetadata

class BrowserPlusPlugin(StaticLibraryPlugin):
    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="BrowserPlus",
            package_name="robotframework-browserplus",
            import_path="BrowserPlus",
            description="Browser wrapper with custom helpers",
            library_type="external",
            categories=["web", "testing"],
            use_cases=["web testing", "visual validation"],
            installation_command="pip install robotframework-browserplus",
            load_priority=7,
        )
        capabilities = LibraryCapabilities(
            contexts=["web"],
            features=["visual-baseline", "grid"],
            supports_page_source=True,
        )
        super().__init__(metadata=metadata, capabilities=capabilities)
```

3. Register the plugin via entry points (in `pyproject.toml`):

```toml
[project.entry-points."robotmcp.library_plugins"]
browser_plus = "my_plugins.browser_plus:BrowserPlusPlugin"
```

4. Publish/install the package. `rf-mcp` will load the plugin automatically on startup.

### Optional hooks

```python
class BrowserPlusPlugin(StaticLibraryPlugin):
    def __init__(self) -> None:
        ...
        self._provider = BrowserPlusStateProvider()

    def get_state_provider(self):
        return self._provider

    def on_session_start(self, session):
        session.variables["BROWSERPLUS_ENABLED"] = True
```

Implement `LibraryStateProvider` or `TypeConversionProvider` if your library requires custom page-source management or argument conversions.

## Quick Start (Manifest-Based Plugin)

Manifest plugins are ideal for workspace-local overrides or staging a new plugin without packaging.

1. Create `.robotmcp/plugins/my_plugin.json` in your workspace:

```json
{
  "metadata": {
    "name": "CustomAPI",
    "package_name": "custom-api-lib",
    "import_path": "CustomAPI",
    "description": "In-house API helpers",
    "library_type": "external",
    "use_cases": ["api testing", "internal systems"],
    "categories": ["api"],
    "contexts": ["api"],
    "installation_command": "pip install custom-api-lib",
    "requires_type_conversion": false,
    "load_priority": 60
  }
}
```

2. Alternatively, point to a Python class:

```json
{
  "module": "examples.plugins.sample_plugin",
  "class": "SamplePlugin"
}
```

3. Start `rf-mcp`; the manifest directory defaults to `.robotmcp/plugins`. Override with the `ROBOTMCP_PLUGIN_PATHS` environment variable (colon-separated paths).

## Testing Plugins

- Unit test plugin metadata and hooks by instantiating your class directly.
- Use `reset_library_plugin_manager_for_tests()` and `library_registry._reset_plugin_state_for_tests()` to clear global caches between tests.
- For integration testing, create an `ExecutionSession` and invoke plugin hooks or state providers as in `tests/test_plugins_basic.py`.

## Debugging & Diagnostics

- `get_library_plugin_manager().list_plugin_names()` lists loaded plugins.
- The plugin manager logs warnings if a plugin fails to load or has incompatible `schema_version`.
- `library_registry.get_all_libraries()` shows the `LibraryConfig` view after plugins are applied.

## Tips

- Use `StaticLibraryPlugin` for metadata-only plugins.
- Keep `schema_version >= 1`; newer API revisions will bump expectations.
- Guard heavy initialization in hooks to avoid slowing down server startup.
- Provide installation hints (`InstallAction`) so MCP tools can surface actionable guidance.
- When supplying page-source providers, return a dict compatible with `PageSourceService` (see `DummyStateProvider` in tests).

See `examples/plugins/sample_plugin/` for a complete runnable example with hooks and manifest integration.

