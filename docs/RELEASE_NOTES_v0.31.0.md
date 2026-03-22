# rf-mcp v0.31.0 Release Notes

**Release Date:** 2026-03-22
**Previous Release:** v0.30.1 (2026-02-26)

---

## Highlights

- **ADR-020: Namespace & Execution Architecture Alignment** - Fixed zombie variable scopes, removed unsafe direct method bypass, improved RF lifecycle correctness
- **BDD & Data-Driven Test Suite Generation** - Full BDD scenario quality improvement with embedded arguments and data-driven template support
- **Persistent Semantic Memory** - sqlite-vec + model2vec based memory system with proactive response augmentation
- **Token Reduction & Tool Optimization** - 71-88% token savings across all tools via artifact externalization, slim profiles, and delta state
- **Copilot CLI E2E Tests** - Full CI integration for AI agent-driven end-to-end testing
- **3 New E2E Test Suites** - Web (Demoshop BDD), API (Restful Booker), XML (Books & Authors)

---

## New Features

### ADR-020: Namespace & Execution Architecture Alignment
- **Fixed zombie variable scope bug** (scope depth 4 instead of correct 3) by removing redundant `namespace.start_test()` / `namespace.end_test()` calls - `ctx.start_test/end_test` handles namespace scope management internally
- **Removed `_try_execute_from_library`** - Direct Python method calls bypassed RF's variable resolution pipeline (`${var}` passed as literal strings). All keyword execution now goes through `runner.run()` or `BuiltIn.run_keyword()`
- **Guarded double variable assignment** - RF's `VariableAssigner` already handles assignment via `RunKeyword.assign`; manual assignment now only runs for the BuiltIn fallback path
- **Added P1 ExecutionStatus guard** in consolidated runner exception handling to prevent BuiltIn retry on assertion failures
- **Initial MCP test tracking** - `create_context_for_session` now stores the initial test in `current_run_test/current_res_test` so `start_test_in_context` can auto-end it
- 16 new lifecycle tests + 3 benchmarks (test cycle 368us, keyword exec 440us, context creation 6.1ms)

### BDD & Data-Driven Test Suite Generation (ADR-019)
- **BDD scenario quality improvement** - 4-phase implementation with word-level embedded name building, instrumentation filtering, semantic phase detection, and content-aware keyword naming
- **Data-driven template support** - `suite_template`, `column_headers`, `named_data_rows` fields; `add_data_row` action with named rows; auto-promotion mode
- **5 critical data-driven bug fixes** - Template body rendering, BDD wrapper argument forwarding, empty data values (`${EMPTY}`), RF builtin variable filtering
- **BDD coverage gap fixes** - 35 new tests, `_build_test_case()` legacy path fix for `bdd_group`/`bdd_intent`

### Persistent Semantic Memory (ADR-014 + ADR-014.2)
- **sqlite-vec + model2vec** based memory system with 3-tier fallback (model2vec -> fastembed -> sentence-transformers)
- **5 MCP tools**: `recall_step`, `recall_fix`, `recall_locator`, `store_knowledge`, `get_memory_status`
- **Proactive response augmentation** - Memory hints injected into execute_step/get_session_state/analyze_scenario responses
- **Natural language step descriptions** for better embedding similarity
- **Memory effectiveness benchmarks** - -6% calls, -7% tokens, -22% time

### Token Reduction & Tool Optimization (ADR-015/016/017/018)
- **ADR-015**: Artifact externalization - 6 domain files + `fetch_artifact` MCP tool
- **ADR-016**: Slim profiles - `SchemaMode`, `SlimToolSchema`, `AutoProfileSelection` for 8K model support
- **ADR-017**: Token accounting - 7 domain files, pluggable backends
- **ADR-018**: Delta state - `mode`/`since_version` params on `get_session_state`
- **Token savings**: 71-88% across all tools; 15-step workflow: 8,763 -> 2,508 tokens

### Output Noise Reduction
- `session_variables` removed from standard/error responses (80% reduction per step)
- `changed_variables` tracking: only non-RF-builtin vars that changed between steps
- Hint deduplication, grouped warnings, compact HTTP responses
- Auto-delta mode for `get_session_state`

### Copilot CLI E2E Tests
- Full CI integration for AI agent-driven end-to-end testing with multi-model support
- Rate limit detection and inter-test delay
- Web, API, and XML test scenarios

### 3 New E2E Test Suites
- **Web**: Demoshop BDD purchase workflow (Browser library, 21 steps, Given/When/Then style)
- **API**: Restful Booker CRUD + authentication (RequestsLibrary, 31 steps, 4 test cases)
- **XML**: Books & Authors parsing (XML library, 33 steps, 4 test cases)

---

## Bug Fixes

- **Fixed zombie variable scope** in RF execution context (ADR-020 F1)
- **Fixed direct method bypass** skipping `${var}` resolution (ADR-020 F3)
- **Fixed double variable assignment** when `RunKeyword.assign` already handles it (ADR-020 F4)
- **Fixed RF context leaks** between test files in CI via autouse cleanup fixtures
- **Fixed leaked mock contamination** - `EXECUTION_CONTEXTS` module-level reference restored to real object when mocks leak
- **Fixed PR #55 review comments** and `requests` import crash
- **Fixed CI failures** from BDD stripping, enum mismatch, and missing E2E fields
- **Fixed Windows SQLite file-lock** in memory integration test teardown
- **Fixed `find_keywords` test** to deny grep/bash/shell and force MCP tool usage
- **5 data-driven bug fixes** (C1-C4, M2, M3): template rendering, BDD wrapper args, empty values, builtin var filtering

---

## CI/CD Improvements

- Copilot CLI-based E2E tests with multi-model support
- Rate limit detection and inter-test delays
- Increased timeouts for autonomous (55min), workflow (300s/15min), job (90min)
- `COPILOT_MODEL` read from GitHub repository variable

---

## Test Suite

| Metric | v0.30.1 | v0.31.0 |
|--------|---------|---------|
| Unit tests | ~5205 | 5209 |
| Total tests (all suites) | ~5900 | 6000+ |
| New test files | - | 4 |
| Benchmarks | - | 3 new (lifecycle) |

---

## Breaking Changes

None. All changes are backward compatible.

---

## Dependencies

No new required dependencies. Optional `memory` extra unchanged (`sqlite-vec` + `model2vec`).

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

## MCP Server Setup

```bash
claude mcp add rf-mcp -- uvx rf-mcp@0.31.0
```
