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
    def get_keyword_library_map(self) -> Optional[Dict[str, str]]: ...
    def get_keyword_overrides(self) -> Optional[Dict[str, KeywordOverrideHandler]]: ...
    def before_keyword_execution(self, session, keyword_name, library_manager, keyword_discovery) -> None: ...
    def on_session_start(self, session: ExecutionSession) -> None: ...
    def on_session_end(self, session: ExecutionSession) -> None: ...
```

Only `get_metadata` is required; every other method is optional.

## Get Started Checklist

- Pick a discovery mechanism: **entry point** for packaged plugins or a **manifest** for workspace-only experiments.
- Implement a plugin class that returns `LibraryMetadata` (and optional hooks such as `LibraryStateProvider` or keyword overrides).
- Register the plugin (entry point in `pyproject.toml` or manifest JSON) and install it into the environment running `rf-mcp` (for example `uv pip install --editable path/to/plugin`).
- Restart `rf-mcp` or call `reload_library_plugins` so the new metadata is loaded.
- Add targeted tests that exercise overrides/state providers (see `tests/unit/test_doctest_plugins.py`).

## Quick Start (Python Package)

Follow these steps to go from an empty plugin to running inside rf-mcp.

### 1. Scaffold the Plugin Class

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

### 2. Register the Plugin

Add an entry point so rf-mcp can discover it automatically:

```toml
[project.entry-points."robotmcp.library_plugins"]
browser_plus = "my_plugins.browser_plus:BrowserPlusPlugin"
```

### 3. Install & Verify

```bash
pip install -e .  # or publish package
```

Start a Python shell:

```python
from robotmcp.config import library_registry
libs = library_registry.get_all_libraries()
print("BrowserPlus" in libs)  # -> True if discovery worked
```

### 4. Run in rf-mcp

Launch the server (`uv run robotmcp`) and use tools such as `recommend_libraries` or `set_library_search_order` to confirm the new library appears. Agents can now execute keywords with your plugin’s overrides/hooks.

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

The DocTest plugins under `examples/plugins/doctest_plugin/rfmcp_doctest_plugin/` show real-world overrides and state providers (visual diff artefacts, PDF comparison summaries, print job metadata, and AI diagnostics).

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
- Use MCP tools `list_library_plugins` and `diagnose_library_plugin` (or the CLI equivalents) to inspect plugin state without restarting the server.
- The plugin manager logs warnings if a plugin fails to load or has incompatible `schema_version`.
- `library_registry.get_all_libraries()` shows the `LibraryConfig` view after plugins are applied.

## Advanced Hooks

- **Keyword Routing**: Return a map from keyword name → library with `get_keyword_library_map()`. Names are normalised to lowercase so both `"Get"` and `"get"` work. This is how the Browser/Selenium/Requests builtin plugins claim specific keywords.
- **Pre-execution Hooks**: Implement `before_keyword_execution()` to prepare state or register RF contexts before a keyword is executed. The Requests plugin uses this to synchronise sessions.
- **Keyword Overrides**: Supply async handlers via `get_keyword_overrides()` when you need to execute a keyword yourself (bypassing default resolution). Return the usual execution payload (`success`, `output`, `error`, `state_updates`, …) to short-circuit normal execution.
- **Application State**: Provide a `LibraryStateProvider` when the plugin can expose domain-specific state. The example below demonstrates how an override writes comparison results into session variables and the state provider surfaces it through MDC agents.

### Example: State Provider + Override

```python
from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import LibraryStateProvider
from robotmcp.components.execution.keyword_executor import ExecutionSession
from robot.libraries.BuiltIn import BuiltIn

class ScreenshotStateProvider(LibraryStateProvider):
    async def get_page_source(self, *args, **kwargs):
        return None  # no DOM

    async def get_application_state(self, session: ExecutionSession):
        result = session.variables.get("_doctest_visual_result")
        if not result:
            return {"success": False, "error": "No visual comparison recorded."}
        return {"success": True, "visual": result}

class DocTestVisualPlugin(StaticLibraryPlugin):
    def __init__(self, metadata, capabilities):
        super().__init__(metadata=metadata, capabilities=capabilities)
        self._state_provider = ScreenshotStateProvider()

    def get_state_provider(self):
        return self._state_provider

    def get_keyword_overrides(self):
        async def override(session, keyword_name, args, keyword_info):
            session.import_library("DocTest.VisualTest", force=False)
            built_in = BuiltIn()
            try:
                built_in.run_keyword(keyword_name, args)
            except AssertionError as exc:
                # Serialise diff artefacts & store in session for later retrieval
                summary = {
                    "status": "failed",
                    "message": str(exc),
                    "artifacts": [{"path": "/tmp/diff.png"}],
                }
                session.variables["_doctest_visual_result"] = summary
                return {
                    "success": False,
                    "output": summary["message"],
                    "error": summary["message"],
                    "state_updates": {"doctest": {"visual": summary}},
                }
            # success path
            summary = {"status": "passed", "message": "Visual comparison passed"}
            session.variables["_doctest_visual_result"] = summary
            return {
                "success": True,
                "output": summary["message"],
                "state_updates": {"doctest": {"visual": summary}},
            }
        return {"compare images": override}
```

- `get_keyword_overrides()` intercepts failures, persists a rich summary in `session.variables`, and returns structured data back to the agent.
- `get_state_provider()` surfaces the latest result through `get_application_state`.
- Attachments could be stored as files (paths) or small base64 strings depending on size constraints.

The production-ready implementation lives in `examples/plugins/doctest_plugin/rfmcp_doctest_plugin/visual.py`. The companion modules (`pdf.py`, `print_jobs.py`, `ai.py`) demonstrate other override patterns and are covered by `tests/unit/test_doctest_plugins.py`.

### Tips

- **Attach Artifacts**: Use temp files (`tempfile.NamedTemporaryFile`) to store images / JSON and pass the file paths in the plugin response. Agents can fetch them via attachments.
- **State Keys**: Namespaced state updates (e.g. `{"doctest": {"visual": summary}}`) avoid collisions with other plugins.
- **Async Hooks**: Keyword overrides are awaited, so you can perform asynchronous work if needed (e.g. call external APIs).
- **MCP Tools**: Document how agents should request the additional state (e.g. via `get_application_state(session_id)` with a `state_type` convention).

## Tips

- Use `StaticLibraryPlugin` for metadata-only plugins.
- Keep `schema_version >= 1`; newer API revisions will bump expectations.
- Guard heavy initialization in hooks to avoid slowing down server startup.
- Provide installation hints (`InstallAction`) so MCP tools can surface actionable guidance.
- When supplying page-source providers, return a dict compatible with `PageSourceService` (see `DummyStateProvider` in tests).

See `examples/plugins/sample_plugin/` for a complete runnable example with hooks and manifest integration.
