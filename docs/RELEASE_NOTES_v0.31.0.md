# rf-mcp v0.31.0

**Previous Release:** v0.30.1 (2026-02-26)

---

## New Features

### Namespace & Execution Architecture Alignment (ADR-020)
- Fixed zombie variable scope bug — scope depth was 4 instead of correct 3 due to redundant `namespace.start_test()` / `end_test()` calls
- Removed `_try_execute_from_library` — direct Python method calls bypassed RF's `${var}` resolution. All keywords now go through `runner.run()` or `BuiltIn.run_keyword()`
- Initial MCP test is now tracked in `current_run_test/current_res_test` so `start_test_in_context` can auto-end it properly

### BDD & Data-Driven Test Suite Generation (ADR-019)
- BDD scenario quality improvement — word-level embedded name building, instrumentation filtering, semantic phase detection, content-aware keyword naming
- Data-driven template support — `suite_template`, `column_headers`, `named_data_rows`, `add_data_row` action with auto-promotion mode
- `bdd_group` / `bdd_intent` params on `execute_step` for behavioral keyword grouping

### Persistent Semantic Memory (ADR-014)
- sqlite-vec + model2vec based memory with 3-tier fallback
- 5 MCP tools: `recall_step`, `recall_fix`, `recall_locator`, `store_knowledge`, `get_memory_status`
- Proactive response augmentation — memory hints injected into tool responses automatically

### Token Reduction (ADR-015/016/017/018)
- Artifact externalization + `fetch_artifact` MCP tool
- Slim profiles with `AutoProfileSelection` for 8K model support
- Delta state via `mode`/`since_version` params on `get_session_state`
- **71-88% token savings** across all tools; 15-step workflow: 8,763 -> 2,508 tokens

### Output Noise Reduction
- `session_variables` removed from responses (80% reduction per step)
- `changed_variables` tracking — only non-builtin vars that changed
- Hint deduplication, grouped warnings, compact HTTP responses, auto-delta mode

---

## Improvements

- Guarded double variable assignment — RF's `VariableAssigner` handles `RunKeyword.assign`; manual assignment now only runs for BuiltIn fallback
- `ExecutionStatus` re-raise guard prevents BuiltIn retry on assertion failures
- Copilot CLI E2E test integration in CI with multi-model support and rate limit detection

---

## Bug Fixes

- Fixed zombie variable scope in RF execution context creating 4 scopes instead of 3
- Fixed direct method bypass skipping `${var}` resolution in keyword arguments
- Fixed double variable assignment when `RunKeyword.assign` already handles it
- Fixed `requests` import crash (PR #55)
- Fixed Windows SQLite file-lock in memory integration test teardown
- Fixed BDD stripping, enum mismatch, and missing E2E fields in CI
- 5 data-driven fixes: template rendering, BDD wrapper args, empty values (`${EMPTY}`), builtin var filtering

---

## Installation

```bash
pip install rf-mcp==0.31.0

# With optional features
pip install rf-mcp[web]==0.31.0      # Browser + SeleniumLibrary
pip install rf-mcp[api]==0.31.0      # RequestsLibrary
pip install rf-mcp[memory]==0.31.0   # Persistent memory
pip install rf-mcp[all]==0.31.0      # Everything
```

```bash
claude mcp add rf-mcp -- uvx rf-mcp@0.31.0
```
