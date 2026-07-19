## 1. Fix v3 tool enumeration

- [x] 1.1 In `ToolManagerCompat.get_tools()` (compat/fastmcp_compat.py), add a v3 branch: if the server exposes an (async) `list_tools`, await it and return `{t.name: t for t in tools}`. Order: existing `get_tools` → `list_tools` → `_tool_manager` → `{}` (with the warning only when truly nothing works).
- [x] 1.2 Tidy the dead `_tool_manager` probe in `remove_tool`/`add_tool` for v3 so they clearly resolve to `server.disable(names=…)` / `server.enable(names=…)`; add `ToolManagerCompat.enable_tool(name)` / `enable_tools(names)` as the name-based v3 primitives.

## 2. Adapter graceful degradation (v3 = name-based only, no churn)

- [x] 2.1 `add_tool_with_description`: on v3, re-enable purely by NAME (`enable_tool`) — never clone + re-register. Works even for tools with no snapshot/descriptor (e.g. `execute_batch`). Per-profile descriptions are not applied on v3 (documented).
- [x] 2.2 `swap_tool_description`: NO-OP on v3. The disable→re-enable churn corrupts FastMCP 3.x's tool-provider chain into unbounded `list_tools()` recursion (`maximum recursion depth exceeded`, reproduced across repeated `browser_exec`/`discovery`/`minimal_exec` → `full` swaps). Since v3 descriptions are immutable per-profile and the tool is already visible, a swap has nothing to do.
- [x] 2.3 `restore_all`: on v3, bulk `enable_tools(all snapshot names)` in one call — no remove-then-add churn.

## 3. Verify (integration gate)

- [x] 3.1 `tests/integration/test_adr_integration.py::TestToolListAfterProfileSwitch::test_tools_reduced_after_small_profile` passes on fastmcp 3.x (count drops after `small_context`).
- [x] 3.2 `tests/unit/test_fastmcp_compat.py` still green (v3 get_tools returns the real set; new v3 no-churn regression tests added).
- [x] 3.3 Full `tests/integration/` suite on 3.x: **598 passed, 0 failed** (baseline was 597 passed + 1 failed = the regression). Excludes the environment-only PlatynUI desktop E2E (which CI skips: PlatynUI not installed on the runner).
- [x] 3.4 Full `tests/unit/` remains green (7052 passed with the new tests).

## 4. Tests

- [x] 4.1 Compat unit test: `ToolManagerCompat.get_tools()` on a v3-shaped server (async `list_tools`, no `_tool_manager`) returns the enumerated `{name: tool}` — not `{}`.
- [x] 4.2 Adapter regression tests (recursion guard): on v3, `swap_tool_description` issues no enable/disable; `add_tool_with_description` enables by name (even with no snapshot); `restore_all` bulk-enables by name — none clone or churn.
