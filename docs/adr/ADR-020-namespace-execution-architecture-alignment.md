# ADR-020: Namespace & Execution Architecture Alignment

**Status:** Implemented
**Date:** 2026-03-20
**Authors:** Automated analysis (4 parallel research agents)
**Supersedes:** —
**Informed by:** [Namespace Architecture Review](../issues/namespace_architecture/rf_mcp_namespace_review.md), 8 reproducible experiments

---

## Context

An external code review of `rf_native_context_manager.py` against Robot Framework 7.3.2 internals revealed 5 findings (1 bug, 2 issues, 2 minor) where rf-mcp's execution lifecycle deviates from RF's native ordering and semantics. All findings were independently reproduced and validated against the installed RF 7.3.2 source code.

### Key Observation

rf-mcp's fundamental architecture — `Namespace` + `EXECUTION_CONTEXTS` + `namespace.get_runner()` → `runner.run()` — is **correct**. The deviations are in lifecycle ordering and redundant/duplicate calls, not in the core approach.

### Scope

All findings reside in a single file: `src/robotmcp/components/execution/rf_native_context_manager.py` (~1200 lines). No changes to server.py, session_manager.py, or other components are required.

---

## Decision

Align rf-mcp's execution lifecycle with RF 7.3.2's native ordering in 3 phases, from highest-impact/lowest-risk to lowest-impact/highest-risk.

---

## Findings Summary

| # | Severity | Finding | Location (current lines) | Fix Effort |
|---|----------|---------|-------------------------|------------|
| F1 | BUG | Double `start_test` / `end_test` creates zombie variable scope | L341-349, L764-765, L802-803 | Remove 3 lines |
| F2 | ISSUE | Lifecycle ordering deviates from RF native | L224-349 (`create_context_for_session`) | Reorder ~15 lines |
| F3 | ISSUE | Direct method bypass skips variable resolution | L674-702 (`_try_execute_from_library`) | Delete ~30 lines |
| F4 | MINOR | Double variable assignment (RF auto-assigns + manual) | L847-850, L1028-1031, L1102-1145 | 2-line guard |
| F5 | MINOR | Double execution path (generic → outer retry) | L826-1044 (`_execute_with_native_resolution`) | Consolidate ~150 lines |

---

## Finding Details

### F1 — BUG: Double `start_test` / `end_test` Creates Zombie Variable Scope

**Root cause:** `_ExecutionContext.start_test()` internally calls `self.namespace.start_test()` (confirmed in RF 7.3.2 source). rf-mcp calls `namespace.start_test()` explicitly *before* `ctx.start_test()`, creating a double push:

```
rf-mcp:     global[0] → suite[1] → ZOMBIE_TEST[2] → test[3]   ← 4 scopes
RF native:  global[0] → suite[1] → test[2]                      ← 3 scopes
```

**Three affected locations:**

| Location | Lines | Pattern |
|----------|-------|---------|
| `create_context_for_session()` | 341-349 | `namespace.start_test()` then `ctx.start_test()` |
| `start_test_in_context()` (ADR-005) | 764-765 | `namespace.start_test()` then `ctx.start_test()` |
| `end_test_in_context()` (ADR-005) | 802-803 | `ctx.end_test()` then `namespace.end_test()` |

**Experimental proof:** Scope depth measured at 4 instead of 3 after context setup. The double push + double pop is *balanced* (4→2 on end), so basic variable operations work. However:

1. **User keyword scope anomaly:** `start_user_keyword()` creates a keyword scope that inherits from the zombie test scope, making test-scoped variables visible inside user keywords where they should not be (confirmed in experiment 7).

2. **TEST-scoped library corruption:** `Namespace.end_test()` calls `lib.scope_manager.end_test()` for every library. `TestScopeManager.end_test()` calls `self.instance_cache.pop()`. Double end_test pops twice — the second pop removes the *suite-level* entry, corrupting the library instance cache. Standard RF libraries (BuiltIn, Browser, SeleniumLibrary) use GLOBAL scope so this is dormant, but any custom TEST-scoped library will break.

**Fix:** Remove redundant calls. Let `ctx.start_test()` / `ctx.end_test()` handle namespace scope management:

```python
# create_context_for_session ~L341:
namespace.start_suite()
# DELETE: namespace.start_test()   ← remove
ctx.start_test(run_test, res_test)  # handles ns.start_test() internally

# start_test_in_context ~L764:
# DELETE: namespace.start_test()   ← remove
ctx.start_test(run_test, res_test)

# end_test_in_context ~L802:
ctx.end_test(res_test)              # handles ns.end_test() internally
# DELETE: namespace.end_test()     ← remove
```

---

### F2 — ISSUE: Lifecycle Ordering Deviates from RF Native

**RF 7.3.2 `SuiteRunner.start_suite()` order** (confirmed via source inspection):

```
1. Create Namespace
2. namespace.start_suite()                    ← push suite variable scope
3. set_from_variable_section()                ← load *** Variables ***
4. EXECUTION_CONTEXTS.start_suite()           ← push context
5. ctx.set_suite_variables()                  ← set ${SUITE_NAME} etc.
6. namespace.handle_imports()                 ← import BuiltIn + Easter + user libs
7. variables.resolve_delayed()                ← resolve cross-referencing variables
```

**rf-mcp's current order** (`create_context_for_session` L121-349):

```
1. Create VariableScopes + manually set ${True} etc.
2. Create TestSuite + Namespace
3. import_library("BuiltIn")                  ← BEFORE context push
4. EXECUTION_CONTEXTS.start_suite()           ← push context
5. Import user libraries
6. namespace.start_suite()                    ← AFTER context push (wrong order)
7. namespace.start_test() + ctx.start_test()  ← immediate (double-push, F1)
```

**Impact analysis:**

| Deviation | Severity | Real impact for MCP use case |
|-----------|----------|------------------------------|
| `import_library("BuiltIn")` before context | Low | Works for BuiltIn (no listeners). Would fail for libraries with listeners (e.g., SeleniumLibrary). rf-mcp wraps in try/except. |
| `namespace.start_suite()` after context | Low | `set_from_variable_section()` never happens, but rf-mcp doesn't parse `.robot` files. |
| Missing `ctx.set_suite_variables()` | Low | `${SUITE_NAME}`, `${SUITE_SOURCE}`, `${SUITE_DOCUMENTATION}` are never set. |
| Missing `resolve_delayed()` | Low | Cross-referencing variables won't resolve. rf-mcp doesn't use `*** Variables ***`. |

**Correction to original review:** The review claims `import_library("BuiltIn")` before context push fails with `AttributeError`. This is **incorrect for BuiltIn** on RF 7.3.2 — BuiltIn has `_has_listeners = False`, so `GlobalScopeManager.start_suite()` → `register_listeners()` returns immediately without accessing `EXECUTION_CONTEXTS.current`. The failure only occurs for listener-bearing libraries (e.g., SeleniumLibrary).

**Fix:** Adopt RF-native ordering:

```python
# 1. Create namespace (existing)
namespace = Namespace(variables, result_suite, suite.resource, Languages())

# 2. Start suite scope FIRST (move before context push)
namespace.start_suite()

# 3. Push context (existing)
ctx = EXECUTION_CONTEXTS.start_suite(result_suite, namespace, output, dry_run=False)
ctx.set_suite_variables(result_suite)       # NEW: set ${SUITE_NAME} etc.

# 4. Import libraries (context exists, so all libraries work)
namespace.handle_imports()                   # BuiltIn + Easter
for lib_name in user_libraries:
    namespace.import_library(lib_name)
namespace.variables.resolve_delayed()        # NEW: resolve cross-refs

# 5. Start test (single call, per F1 fix)
ctx.start_test(run_test, res_test)
```

---

### F3 — ISSUE: Direct Method Bypass Skips Variable Resolution

**Location:** `_try_execute_from_library()` (L674-702), called from `_execute_any_keyword_generic()` (L647-665).

**Problem:** Falls back to calling Python methods directly on library instances:

```python
method_name = keyword_name.replace(' ', '_').lower()
method = getattr(lib_instance, method_name)
return method(*arguments)       # ← raw call, NO RF variable resolution
```

This bypasses RF's entire argument processing pipeline: variable resolution (`${var}` expansion), type conversion, named argument parsing, output capture, and status reporting.

**Experimental proof:**
```
Through RF runner:  bi.run_keyword("Set Variable", "Hello ${name}") → "Hello World"
Direct method call: bi.set_variable("Hello ${name}")                → "Hello ${name}"
```

**Reachability:** The bypass is only reached when `namespace.get_runner()` raises a non-`ExecutionStatus` exception in path 1 of `_execute_any_keyword_generic()` — which experiments confirmed is **very rare**. `get_runner()` returns `InvalidKeywordRunner` for unknown keywords (no exception). `runner.run()` raises `ExecutionStatus` for all standard failure modes (wrong args, keyword not found, assertion failures), which rf-mcp already correctly catches and re-raises.

**Fix:** Remove `_try_execute_from_library()` entirely (~30 lines). The resolution order becomes:

```
1. namespace.get_runner() → runner.run()     (RF native, correct)
2. BuiltIn.run_keyword() fallback             (existing, correct)
```

---

### F4 — MINOR: Double Variable Assignment

**Problem:** When `runner.run()` is called with `assign=("${var}",)` on the `RunKeyword` model, RF's `VariableAssigner.assign()` automatically writes the return value into `variables[name]`. Then `_handle_variable_assignment()` writes the **same value again**.

**Confirmed via RF source:**
```python
# LibraryKeywordRunner.run():
with assignment.assigner(context) as assigner:
    return_value = self._run(data, kw, context)
    assigner.assign(return_value)         # ← RF auto-assigns here
```

**Two redundant call sites:**
- L847-850: After `_execute_any_keyword_generic()` succeeds
- L1028-1031: After outer `runner.run()` succeeds

Both paths build `RunKeyword` with `assign=tuple(assign_to)` at L965, so RF already handles the assignment.

**Impact:** Harmless — overwrites the same value. The only cost is a redundant dictionary write.

**Fix:** Guard manual assignment:

```python
# Skip _handle_variable_assignment when RunKeyword.assign was set
# (RF already handled it via VariableAssigner)
if assign_to and result is not None and not data_kw.assign:
    assigned_vars = self._handle_variable_assignment(assign_to, result, variables)
```

---

### F5 — MINOR: Double Execution Path (Generic → Outer) is Wasteful

**Problem:** `_execute_with_native_resolution()` (L826-1044) calls `_execute_any_keyword_generic()` first (L842), which tries `namespace.get_runner()` → `runner.run()`. If this fails with a non-`ExecutionStatus` exception, it falls through to **another** `namespace.get_runner()` → `runner.run()` at L953.

The outer path does the same work as the inner path, plus:
- Builds `RunKeyword` with `assign=` parameter
- Sets `res_kw.parent` for StatusReporter
- Has evaluate-expression normalization (L884-945)

**Impact:** ~150 lines of nested complexity. The outer path is reached when the inner path raises a non-`ExecutionStatus` exception, which is rare (confirmed by experiment 4). The outer path adds the `assign=` and `parent` context that the inner path lacks — so the inner path is actually **less correct** than the outer.

**Fix:** Consolidate into a single try/except chain. The outer path should be the *primary* path (it sets `assign=` and `parent`), with `BuiltIn.run_keyword` as the only fallback:

```python
def _execute_with_native_resolution(self, session_id, keyword_name, arguments,
                                      namespace, variables, assign_to=None):
    # Single resolution path with proper models
    runner = namespace.get_runner(keyword_name)
    data_kw = RunKeyword(name=..., args=..., assign=tuple(assign_to) or ())
    res_kw = ResultKeyword(name=..., parent=parent_test)

    try:
        result = runner.run(data_kw, res_kw, ctx)
    except ExecutionStatus:
        raise
    except Exception:
        result = BuiltIn().run_keyword(keyword_name, *arguments)

    return {"success": True, "result": result, ...}
```

---

## DDD Domain Model

### Bounded Context: `execution`

The `rf_native_context_manager.py` module is part of the **Execution** bounded context. The proposed changes affect 3 aggregates:

```
execution/
├── aggregates/
│   ├── rf_context_lifecycle.py       ← F1 + F2: Suite/test lifecycle ordering
│   ├── keyword_dispatcher.py         ← F3 + F4 + F5: Keyword execution pipeline
│   └── variable_scope_manager.py     ← F1 + F4: Variable scope management
├── value_objects/
│   ├── lifecycle_phase.py            ← Enum: SUITE_START, TEST_START, TEST_END, SUITE_END
│   ├── execution_order.py            ← Tuple defining correct RF lifecycle sequence
│   └── scope_depth.py                ← Value object tracking expected vs actual scope count
├── events/
│   ├── suite_lifecycle_started.py    ← Raised after correct suite setup
│   ├── test_scope_pushed.py          ← Raised after ctx.start_test (single push)
│   ├── test_scope_popped.py          ← Raised after ctx.end_test (single pop)
│   └── keyword_executed.py           ← Raised after runner.run() completes
└── services/
    ├── lifecycle_orchestrator.py      ← Ensures RF-native ordering of create_context
    └── keyword_resolution_service.py  ← Single-path keyword dispatch
```

### Aggregate: `RFContextLifecycle` (F1 + F2)

**Invariants:**
1. `namespace.start_test()` is called exactly once per test start (via `ctx.start_test()`)
2. `namespace.end_test()` is called exactly once per test end (via `ctx.end_test()`)
3. Suite setup follows RF-native ordering: namespace → start_suite → context push → imports → start_test
4. Variable scope depth after `start_test` = base + 1 (never base + 2)

### Aggregate: `KeywordDispatcher` (F3 + F4 + F5)

**Invariants:**
1. Keywords always execute through RF's native pipeline (`runner.run()`)
2. No direct Python method calls bypass variable resolution
3. Variable assignment happens exactly once per keyword execution
4. Single execution attempt per keyword (no retry of the same mechanism)

---

## Test Gap Analysis

The following test gaps were identified by the analysis agents. These must be filled **before** implementing fixes:

### Critical Gaps (must add before F1 fix)

| Gap | Description | Risk if unfilled |
|-----|-------------|------------------|
| GAP-1 | **Test cycling with real RF context**: No tests verify actual `VariableScopes._scopes` depth during `start_test_in_context` / `end_test_in_context` | Fix could silently break scope isolation |
| GAP-2 | **Variable scope transitions**: No tests verify test-scoped variables are cleaned up after `end_test_in_context` | Variable leaks between tests would go undetected |
| GAP-3 | **TEST-scoped library safety**: No tests with TEST-scoped libraries | The F1 fix is specifically motivated by TEST-scope corruption |

### Important Gaps (should add before F5 fix)

| Gap | Description | Risk if unfilled |
|-----|-------------|------------------|
| GAP-4 | **Native resolution end-to-end**: No tests for `_execute_with_native_resolution()` complete flow | Consolidation could break edge cases |
| GAP-5 | **Variable resolution through runner**: No tests verify `${var}` expansion in arguments | Direct bypass removal (F3) untestable |
| GAP-6 | **Multi-test search order**: No tests verify search order is maintained across test boundaries | Search order could drift between tests |

### Existing Coverage (safe to rely on)

| Component | Test File | Tests | Status |
|-----------|-----------|-------|--------|
| Multi-test registry | `test_multi_test_session.py` | 40 | Solid |
| Keyword resolution (C1/C2) | `test_keyword_resolution_fixes.py` | 33 | Solid |
| Search order sync | `test_search_order_and_error_surfacing.py` | 30 | Solid |
| Pre-validation | `test_prevalidation_fixes.py` | 115 | Solid |
| Context creation basics | `test_rf_native_context_manager_context.py` | 2 | Minimal |

---

## Implementation Plan

### Phase 0: Test Scaffolding (prerequisite)

**Goal:** Fill critical test gaps before making any code changes.

| Task | Tests to add | Target file |
|------|-------------|-------------|
| T0.1 | Verify scope depth = 3 after `start_test_in_context` with real RF context | `tests/unit/test_namespace_lifecycle.py` |
| T0.2 | Verify test variables cleaned after `end_test_in_context` | Same file |
| T0.3 | Verify suite variables persist across test boundaries | Same file |
| T0.4 | Multi-test cycling (5 tests) with variable isolation | Same file |
| T0.5 | TEST-scoped library mock: verify `instance_cache` not corrupted | Same file |
| T0.6 | Verify `${var}` resolution through `runner.run()` vs direct call | `tests/unit/test_keyword_dispatch.py` |
| T0.7 | Verify single execution per keyword (no double `runner.run()`) | Same file |
| T0.8 | Variable assignment: verify RF auto-assign when `assign=` is set | Same file |

**Expected:** All T0 tests **fail** on current code for the findings they target (especially T0.1 scope depth = 4 instead of 3), confirming the bugs exist. Tests for correct behavior (T0.2-T0.4) should pass because the double push/pop is balanced.

### Phase 1: F1 Fix — Remove Double start_test/end_test (HIGH PRIORITY)

**Scope:** 3 line deletions + reorder 1 line
**Risk:** Low (balanced push/pop means removing both sides is safe)
**Rollback:** Re-add the 3 lines

| Step | Change | Lines |
|------|--------|-------|
| 1.1 | In `create_context_for_session`: move `namespace.start_suite()` before `ctx = EXECUTION_CONTEXTS.start_suite(...)` | ~L335-341 |
| 1.2 | In `create_context_for_session`: delete `namespace.start_test()` at L342 | L342 |
| 1.3 | In `start_test_in_context`: delete `namespace.start_test()` at L764 | L764 |
| 1.4 | In `end_test_in_context`: delete `namespace.end_test()` at L803 | L803 |

**Validation:** T0.1 (scope depth) changes from 4→3. All 5205 existing tests must pass.

### Phase 2: F3 Fix — Remove Direct Method Bypass (MEDIUM PRIORITY)

**Scope:** Delete `_try_execute_from_library()` (~30 lines) + remove call site in `_execute_any_keyword_generic()` (~15 lines)
**Risk:** Low (bypass is rarely reached; removing it falls through to BuiltIn.run_keyword which is correct)

| Step | Change | Lines |
|------|--------|-------|
| 2.1 | Delete `_try_execute_from_library()` method | L674-702 |
| 2.2 | Remove path 2 (direct method lookup) from `_execute_any_keyword_generic()` | L647-665 |

**Validation:** T0.6 (variable resolution) confirms `${var}` is always expanded. All existing tests pass.

### Phase 3: F2 + F4 + F5 — Lifecycle Reorder + Consolidation (LOWER PRIORITY)

**Scope:** Reorder `create_context_for_session()` (~15 lines), add assignment guard (~2 lines), consolidate execution paths (~150 lines reduced)
**Risk:** Medium (reordering imports could affect library initialization for edge-case libraries; consolidation touches the main execution hot path)

| Step | Change | Lines |
|------|--------|-------|
| 3.1 | Reorder `create_context_for_session()` to RF-native sequence | L224-349 |
| 3.2 | Add `ctx.set_suite_variables(result_suite)` after context push | New |
| 3.3 | Add `namespace.variables.resolve_delayed()` after imports | New |
| 3.4 | Guard `_handle_variable_assignment()` with `if not data_kw.assign` | L847, L1028 |
| 3.5 | Consolidate `_execute_with_native_resolution()`: remove inner generic call, make outer path primary | L826-1044 |
| 3.6 | Optionally delete `_execute_any_keyword_generic()` entirely | L594-672 |

**Validation:** All tests pass. Token accounting baselines may need updating (response structure changes).

---

## Risk Assessment

### What Could Go Wrong

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| F1 fix breaks scope isolation | Very Low | High | Phase 0 tests verify scope depth before/after |
| F2 reorder breaks library import | Low | Medium | Only affects listener-bearing libraries; test with Browser + SeleniumLibrary |
| F3 removal breaks a keyword that ONLY works via direct call | Very Low | Medium | grep for `_try_execute_from_library` in logs; monitor e2e tests |
| F5 consolidation breaks Evaluate normalization | Low | Medium | Evaluate normalization code (L884-945) must be preserved in consolidated path |
| RF version upgrade changes internal APIs | Medium | High | Pin RF version in CI; test against 7.3.x and 7.4.x |

### What Will NOT Break

- **Session management**: No changes to session_manager.py or manage_session actions
- **Build test suite**: No changes to test generation code
- **Page source / DOM**: No changes to page_source_service.py
- **Memory / token optimization**: No changes to ADR-014/015/016/017/018 domains
- **BDD / data-driven**: No changes to ADR-019 keyword resolution domain

---

## Metrics

### Before (Current State)

| Metric | Value |
|--------|-------|
| Variable scope depth after start_test | 4 (incorrect) |
| Execution paths per keyword | 5 (generic → direct → BuiltIn → outer runner → outer BuiltIn) |
| Variable assignments per keyword with assign_to | 2-3 (RF auto + manual + session sync) |
| `_execute_with_native_resolution` complexity | ~220 lines with 3 nested try/except |
| `_execute_any_keyword_generic` lines | 80 |
| `_try_execute_from_library` lines | 30 |

### After (Post-Implementation)

| Metric | Value |
|--------|-------|
| Variable scope depth after start_test | 3 (correct) |
| Execution paths per keyword | 2 (runner.run → BuiltIn fallback) |
| Variable assignments per keyword with assign_to | 1 (RF auto only) |
| `_execute_with_native_resolution` complexity | ~80 lines, single try/except |
| Lines removed | ~200 |

---

## RF Version Compatibility

| RF Version | Verified | Notes |
|-----------|---------|-------|
| 7.3.2 | Yes (installed) | All experiments reproduced |
| 7.4.x | No | Review document referenced 7.4.2 but installed version is 7.3.2. `ctx.start_test()` calling `ns.start_test()` is present in both. Module path differs: `robot.running.librarykeywordrunner` (7.3.2) vs `robot.running.handlers` (claimed in review). |

**Recommendation:** Verify Phase 1 fix against RF 7.4.x before merging. The internal APIs used (`_ExecutionContext.start_test`, `Namespace.start_test`, `VariableScopes._scopes`) are private and may change.

---

## Appendix A: Experiment Reproduction Results

All 8 experiments reproduced on RF 7.3.2. Full outputs available in the experiment tarball.

| # | Experiment | Claimed Finding | Reproduced? | Divergence |
|---|-----------|-----------------|-------------|------------|
| 1 | Ordering | rf-mcp order differs from RF native | CONFIRMED | — |
| 2 | Double Exec | Generic-then-outer causes double exec | CONFIRMED (nuanced) | No actual double exec if generic succeeds |
| 3 | Double Assign | runner.run() auto-assigns; rf-mcp duplicates | CONFIRMED | `import_library("BuiltIn")` succeeds without context (diverges from review) |
| 4 | Bypass Reach | Direct method bypass is rarely reachable | CONFIRMED | Missing parent on ResultKeyword does NOT cause error (diverges from review) |
| 5 | Double Scope | Double start_test → 4 scopes instead of 3 | CONFIRMED | — |
| 6 | ADR-005 | Multi-test cycling has same double-call bug | CONFIRMED | Push/pop is balanced, scopes correct at end |
| 7 | Zombie Scope | Zombie scope affects variable visibility | CONFIRMED | User keyword scope sees test variables through zombie (anomalous) |
| 8 | TEST-Scope | Double end_test corrupts TEST-scoped libs | CONFIRMED (theoretical) | Only affects custom TEST-scoped libs, not standard ones |

### Key Corrections to Original Review

1. **RF version is 7.3.2**, not 7.4.2. Module `robot.running.handlers` does not exist; the keyword runner lives in `robot.running.librarykeywordrunner`.

2. **`import_library("BuiltIn")` does NOT fail without context** on RF 7.3.2. BuiltIn has `_has_listeners = False`, so `GlobalScopeManager.register_listeners()` returns immediately. The failure only occurs for listener-bearing libraries (e.g., SeleniumLibrary).

3. **`Namespace.end_test()` sets `self._running_test = True`** (not `False`). This is intentional RF behavior — the flag governs whether newly imported libraries get `scope_manager.start_test()` during import.

4. **Missing parent on ResultKeyword** does NOT cause a non-`ExecutionStatus` exception. `runner.run()` works fine without it on RF 7.3.2.

---

## Appendix B: RF 7.3.2 Source Verification

### `_ExecutionContext.start_test()` — calls `namespace.start_test()` CONFIRMED

```python
def start_test(self, data, result):
    self.test = result
    self._add_timeout(result.timeout)
    self.namespace.start_test()           # ← HERE
    self.variables.set_test("${TEST_NAME}", result.name)
    # ...
```

### `_ExecutionContext.end_test()` — calls `namespace.end_test()` CONFIRMED

```python
def end_test(self, test):
    self.test = None
    self._remove_timeout(test.timeout)
    self.namespace.end_test()             # ← HERE
    self.variables.set_suite("${PREV_TEST_NAME}", test.name)
    # ...
```

### `Namespace.start_test()` — pushes variable scope CONFIRMED

```python
def start_test(self):
    self._running_test = True
    self.variables.start_test()           # → _scopes.append(new_scope)
    for lib in self.libraries:
        lib.scope_manager.start_test()
```

### `Namespace.end_test()` — pops variable scope + calls lib.scope_manager.end_test() CONFIRMED

```python
def end_test(self):
    self.variables.end_test()             # → _scopes.pop()
    for lib in self.libraries:
        lib.scope_manager.end_test()
    self._running_test = True             # NOTE: stays True (not False)
```

### `TestScopeManager` — instance_cache.pop() CONFIRMED

```python
class TestScopeManager(SuiteScopeManager):
    def start_test(self):
        self.unregister_listeners()
        self.instance_cache.append(self.library._instance)
        self.library.instance = None
        self.register_listeners()

    def end_test(self):
        self.unregister_listeners(close=True)
        self.library.instance = self.instance_cache.pop()   # ← IndexError on double pop
        self.register_listeners()
```

### `SuiteRunner.start_suite()` — canonical lifecycle ordering CONFIRMED

```python
def start_suite(self, data):
    ns = Namespace(self.variables, result, data.resource, ...)
    ns.start_suite()                                              # 1
    ns.variables.set_from_variable_section(data.resource.variables)  # 2
    EXECUTION_CONTEXTS.start_suite(result, ns, self.output, ...)  # 3
    self.context.set_suite_variables(result)                      # 4
    ns.handle_imports()                                           # 5
    ns.variables.resolve_delayed()                                # 6
```

### `LibraryKeywordRunner.run()` — auto-assigns via VariableAssignment CONFIRMED

```python
def run(self, data, result, context, run=True):
    kw = self.keyword.bind(data)
    assignment = VariableAssignment(data.assign)
    with StatusReporter(data, result, context, run, implementation=kw):
        if run:
            with assignment.assigner(context) as assigner:
                return_value = self._run(data, kw, context)
                assigner.assign(return_value)         # ← auto-assigns
                return return_value
```
