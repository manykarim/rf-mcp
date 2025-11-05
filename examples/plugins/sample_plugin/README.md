# Sample Plugin: ExampleCalculatorLibrary

This directory shows a minimal plugin that advertises a fictitious `ExampleCalculatorLibrary` to `rf-mcp`.

## Files

- `plugin.py` – Implements `ExampleCalculatorPlugin`, returning metadata, install actions, session hooks, and a custom `LibraryStateProvider`.
- `manifest.json` – Manifest-based registration pointing at the plugin class.
- `__init__.py` – Exports the plugin for entry-point usage.

## Using the Plugin Locally

1. Copy (or symlink) the manifest into your workspace plugin directory:

```bash
mkdir -p .robotmcp/plugins
cp examples/plugins/sample_plugin/manifest.json .robotmcp/plugins/example-calculator.json
```

2. Launch `rf-mcp`. The plugin manager will load the manifest and register the library. You can verify via:

```python
from robotmcp.config import library_registry
print("Calculator" in library_registry.get_all_libraries())
```

3. Optionally extend the plugin or connect it to a real library by replacing metadata and hooks.

## Packaging Example

If you want to distribute the plugin as a Python package, add an entry point to your `pyproject.toml`:

```toml
[project.entry-points."robotmcp.library_plugins"]
example_calculator = "examples.plugins.sample_plugin:ExampleCalculatorPlugin"
```

Installing that package in the same environment as `rf-mcp` makes the plugin available automatically.

