# rf-mcp v0.30.1 Release Notes

## FastMCP 3.x Compatibility

This patch release adds full compatibility with **fastmcp 3.0+** while retaining support for fastmcp 2.8+.

### Problem

FastMCP 3.0.0 removed the `enabled` parameter from `@mcp.tool()`, causing `TypeError` on startup when rf-mcp was installed alongside fastmcp v3. The `execute_step.fn` accessor and `_mcp_server.lifespan` mutation also changed in v3, breaking internal tool calls and lifespan setup.

### What Changed

- **New compatibility module** (`robotmcp.compat.fastmcp_compat`) — version detection at import time with graceful fallbacks for both v2 and v3 APIs.
- **36 `enabled=False` decorators** replaced with `**DISABLED_TOOL_KWARGS` — expands to `{'enabled': False}` on v2 and `{}` on v3.
- **`finalize_disabled_tools()`** — called once after all tool registrations; on v3 it calls `mcp.disable(names=...)` to hide discovery-only tools from `tools/list`.
- **`get_tool_fn()`** — replaces 3 direct `.fn` accesses; returns the underlying async callable on both v2 (`FunctionTool.fn`) and v3 (plain function).
- **`set_server_lifespan()`** — replaces 2 direct `_mcp_server.lifespan` mutations with version-aware attribute resolution.
- **`ToolManagerCompat`** — abstracts v2 `_tool_manager` private API vs v3 public API for the ADR-006 tool profile system.
- **Tool profile fix** — `get_visible_tool_names()` now filters out disabled tools, preventing them from appearing in profile diffs.
- **Windows CI fix** — AST-based consistency tests now use explicit `encoding="utf-8"` for `pathlib.read_text()` to avoid `cp1252` decode errors.
- **Minimum version pin** — `pyproject.toml` requires `fastmcp>=2.8.0` (first version with `enabled=` support).

### Testing

- 24 new unit tests covering version detection, disabled tool kwargs, `finalize_disabled_tools()`, `get_tool_fn()`, `ToolManagerCompat`, and `set_server_lifespan()`.
- Full test suite passes on both fastmcp 2.x and 3.x.

### Upgrade Notes

No breaking changes. Existing installations on fastmcp 2.8+ continue to work unchanged. Upgrading to fastmcp 3.x now works without modification.
