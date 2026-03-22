# rf-mcp Namespace & Execution Code Review

## Analysis of `rf_native_context_manager.py` against RF 7.4.2 Internals

**Package version analyzed:** rf-mcp 0.14.0 · Robot Framework 7.4.2

---

## Executive Summary

rf-mcp's `RobotFrameworkNativeContextManager` successfully creates a usable RF execution context and can execute keywords stepwise. The fundamental architecture — creating a `Namespace`, pushing an `_ExecutionContext`, and dispatching keywords via `namespace.get_runner()` + `runner.run()` — is correct. However, the implementation deviates from RF's native lifecycle ordering in several ways, some harmless, some problematic. All findings below are backed by reproducible experiments.

### Severity Legend

| Level | Meaning |
|-------|---------|
| 🔴 **BUG** | Incorrect behavior, causes real issues in specific scenarios |
| 🟡 **ISSUE** | Deviation from RF semantics, may cause subtle problems |
| 🟢 **MINOR** | Inefficiency or unnecessary code, functionally benign |
| ✅ **CORRECT** | Working as intended |

---

## Finding 1 — 🔴 Double `start_test` / `end_test` Creates Zombie Variable Scope

### The Problem

In `create_context_for_session()` (lines 341–349), rf-mcp calls:

```python
namespace.start_test()                    # line 342 — pushes test scope
ctx.start_test(run_test, res_test)        # line 349 — calls ns.start_test() AGAIN
```

But `_ExecutionContext.start_test()` internally calls `self.namespace.start_test()`, which means `namespace.start_test()` is called **twice**, creating **4 variable scopes** instead of the correct 3:

```
rf-mcp:     global[0] → suite[1] → ZOMBIE_TEST[2] → test[3]     ← 4 scopes
RF native:  global[0] → suite[1] → test[2]                       ← 3 scopes
```

### Experimental Proof

```
=== rf-mcp lifecycle scope trace ===
  [init]                scopes=1
  [ctx.start_suite]     scopes=1
  [ns.start_suite()]    scopes=2
  [ns.start_test()]     scopes=3      ← extra push
  [ctx.start_test()]    scopes=4      ← internal ns.start_test() pushes again

=== Correct RF lifecycle ===
  [ns.start_suite()]    scopes=2
  [ctx.start_test()]    scopes=3      ← only ONE push
```

The same pattern exists in `start_test_in_context()` (ADR-005, lines 764–765):

```python
namespace.start_test()                    # line 764 — extra push
ctx.start_test(run_test, res_test)        # line 765 — pushes again internally
```

And `end_test_in_context()` (lines 802–803) mirrors the double pop:

```python
ctx.end_test(res_test)                    # line 802 — calls ns.end_test → pops once
namespace.end_test()                      # line 803 — pops AGAIN
```

### Impact

The double push + double pop is **balanced** (4→2 on end), so basic variable operations work. However:

1. **During execution**, the zombie test scope sits between suite and real test scopes. `Set Test Variable` only writes to the **innermost** test scope (scope[3]), not the zombie (scope[2]). This is benign for normal usage.

2. **TEST-scoped library corruption**: `namespace.end_test()` calls `lib.scope_manager.end_test()` for every library. `ctx.end_test()` does the same internally. For `TestScopeManager`, `end_test()` calls `self.instance_cache.pop()`. The **double pop** would corrupt the instance cache for any TEST-scoped library, potentially returning the wrong library instance or crashing with an `IndexError`.

   Standard RF libraries (BuiltIn, Browser, SeleniumLibrary, Collections) all use GLOBAL or SUITE scope, so this hasn't surfaced in practice. But any custom library with `ROBOT_LIBRARY_SCOPE = "TEST"` will break.

### Fix

Remove the redundant calls. Let `ctx.start_test()` / `ctx.end_test()` handle namespace scope management (as RF does natively):

```python
# create_context_for_session, around line 341:
# DELETE: namespace.start_test()                    ← remove this
ctx.start_test(run_test, res_test)                  # this handles ns.start_test() internally

# start_test_in_context, around line 764:
# DELETE: namespace.start_test()                    ← remove this
ctx.start_test(run_test, res_test)

# end_test_in_context, around line 802:
ctx.end_test(res_test)                              # this handles ns.end_test() internally
# DELETE: namespace.end_test()                      ← remove this
```

---

## Finding 2 — 🟡 Lifecycle Ordering Deviates from RF Native

### The Problem

RF's `SuiteRunner.start_suite()` follows this precise order:

```
1. Create Namespace
2. namespace.start_suite()              ← push suite variable scope
3. set_from_variable_section()          ← load *** Variables ***
4. EXECUTION_CONTEXTS.start_suite()     ← push context
5. ctx.set_suite_variables()            ← set ${SUITE_NAME}, ${SUITE_SOURCE} etc.
6. namespace.handle_imports()           ← import BuiltIn, Easter, user libraries
7. variables.resolve_delayed()          ← resolve cross-referencing variables
```

rf-mcp's `create_context_for_session()` does:

```
1. Create VariableScopes + manually set ${True} etc.
2. Create TestSuite + Namespace
3. import_library("BuiltIn")            ← BEFORE context push (can fail!)
4. EXECUTION_CONTEXTS.start_suite()     ← push context
5. namespace.start_suite()              ← AFTER context push (wrong order)
6. namespace.start_test()               ← immediate (double-push, see Finding 1)
7. ctx.start_test()                     ← double-push
8. Import user libraries
```

### Experimental Proof

```python
# import_library("BuiltIn") calls _current_owner_and_lineno()
# which accesses EXECUTION_CONTEXTS.current.steps
# If no context exists yet → AttributeError

namespace.import_library("BuiltIn")   # before context
# → AttributeError: 'NoneType' object has no attribute 'steps'

# But handle_imports() uses _import_library (internal), which does NOT
# call _current_owner_and_lineno for default libraries:
namespace.handle_imports()            # before context → WORKS
```

The `import_library("BuiltIn")` call at rf-mcp line 226 will fail with `AttributeError` if no context exists. rf-mcp wraps it in `try/except` (lines 228-230), so the failure is silently swallowed, and BuiltIn gets imported later via `handle_imports()` or the retry logic. This works but is fragile.

### Impact

- **`namespace.start_suite()` after context push**: This means `set_from_variable_section()` never happens for any `*** Variables ***` defined on the suite. In practice, rf-mcp doesn't parse `.robot` files at this stage, so the section is empty. No real impact for the MCP use case.

- **Missing `ctx.set_suite_variables()`**: `${SUITE_NAME}`, `${SUITE_SOURCE}`, `${SUITE_DOCUMENTATION}` are never set in the standard way. rf-mcp manually sets `${OUTPUTDIR}` etc., but RF's standard suite variables are missing.

- **Missing `resolve_delayed()`**: Variables that reference other variables won't be resolved. Again, rf-mcp doesn't use `*** Variables ***` sections, so no impact.

### Fix

Adopt the RF-native ordering:

```python
# 1. Create namespace
namespace = Namespace(variables, result_suite, suite.resource, Languages())

# 2. Start suite scope FIRST
namespace.start_suite()
namespace.variables.set_from_variable_section(suite.resource.variables)

# 3. Push context
ctx = EXECUTION_CONTEXTS.start_suite(result_suite, namespace, output, dry_run=False)
ctx.set_suite_variables(result_suite)

# 4. Import libraries (context exists, so import_library works)
namespace.handle_imports()           # BuiltIn + Easter
for lib_name in user_libraries:
    namespace.import_library(lib_name)
namespace.variables.resolve_delayed()

# 5. Start test (only via ctx, not double)
ctx.start_test(run_test, res_test)
```

---

## Finding 3 — 🟡 Direct Method Bypass Skips Variable Resolution

### The Problem

`_try_execute_from_library()` (line 674) falls back to calling Python methods directly on library instances:

```python
method_name = keyword_name.replace(' ', '_').lower()
if hasattr(lib_instance, method_name):
    method = getattr(lib_instance, method_name)
    return method(*arguments)           # ← raw call, no RF machinery
```

This bypasses RF's entire argument processing pipeline: variable resolution, type conversion, named argument parsing, output capture, and status reporting.

### Experimental Proof

```python
# Through RF runner (proper):
bi.run_keyword("Set Variable", "Hello ${name}")
# → "Hello World" (variable resolved)

# Direct method call (rf-mcp fallback):
bi.set_variable("Hello ${name}")
# → "Hello ${name}" (no resolution!)
```

Additionally, RF keyword names don't always map to `method_name.lower()`. Some libraries use decorators (`@keyword`), custom names, or method names that don't follow the simple convention. The experiment also showed that `Collections` doesn't have `get_length` — the method is named differently due to RF's internal method discovery.

### Impact

This fallback is **rarely reached** in practice. `_execute_any_keyword_generic()` tries the bypass only if `namespace.get_runner()` raises a non-`ExecutionStatus` exception, which experiments showed is uncommon. However, if it IS reached, the keyword will execute without variable resolution — `${variables}` in arguments will be passed as literal strings.

### Fix

Remove the direct method fallback entirely. If `namespace.get_runner() + runner.run()` fails with `ExecutionStatus`, re-raise immediately (which rf-mcp already does correctly). For any other exception, fall through to `BuiltIn.run_keyword()` which properly resolves variables. The direct method path adds complexity without benefit:

```python
def _execute_any_keyword_generic(self, keyword_name, arguments, namespace):
    # 1. RF-native resolution (correct)
    runner = namespace.get_runner(keyword_name)
    data_kw = RunKeyword(name=keyword_name, args=tuple(arguments))
    res_kw = ResultKeyword(name=keyword_name, args=tuple(arguments))
    ctx = EXECUTION_CONTEXTS.current
    return runner.run(data_kw, res_kw, ctx)
    
    # If ExecutionStatus → re-raised by existing isinstance check
    # If other error → fall through to BuiltIn.run_keyword (existing fallback)
    # DELETE: _try_execute_from_library path entirely
```

---

## Finding 4 — 🟢 Double Variable Assignment (Harmless)

### The Problem

When `runner.run()` is called with `assign=("${var}",)` on the `RunKeyword`, RF's `VariableAssignment` mechanism automatically assigns the return value. Then rf-mcp's `_handle_variable_assignment()` assigns it **again**.

### Experimental Proof

```
=== Does runner.run() auto-assign? ===
  WITH assign: ${result_a} = ['a', 'b', 'c']  (RF auto-assigned ✓)
  
  DOUBLE ASSIGN TEST:
  After runner.run() with assign: ${result_c} = ['1', '2']
  After manual overwrite: ${result_c} = OVERWRITTEN
  => Double assignment causes overwrite (harmless but wasteful)
```

### Impact

The second assignment overwrites the first with the same value. Functionally benign — the only cost is a redundant dictionary write.

### Fix (Optional)

Skip `_handle_variable_assignment()` when `assign` was passed to `RunKeyword`:

```python
# In _execute_with_native_resolution:
data_kw = RunKeyword(name=..., args=..., assign=tuple(assign_to) if assign_to else ())

# If RF handled the assignment via RunKeyword.assign, skip manual assignment
if not data_kw.assign and assign_to and result is not None:
    assigned_vars = self._handle_variable_assignment(assign_to, result, variables)
```

---

## Finding 5 — 🟢 Double Execution Path (Generic → Outer) is Wasteful but Safe

### The Problem

`_execute_with_native_resolution()` calls `_execute_any_keyword_generic()` first (line 842), which tries `namespace.get_runner() + runner.run()`. If this succeeds, it returns immediately. If it fails with a non-`ExecutionStatus` exception, it falls through to **another** `namespace.get_runner() + runner.run()` call (line 953).

This means the same keyword lookup and execution is attempted twice via different code paths.

### Experimental Proof

```
get_runner() returns InvalidKeywordRunner for unknown keywords (not an exception).
runner.run() raises ExecutionStatus for all normal failure modes.
rf-mcp correctly catches ExecutionStatus and re-raises.
```

The outer path (line 953) is only reached if the inner path raises something other than `ExecutionStatus` — which is rare. When it IS reached, the outer path does the same thing: `get_runner() + runner.run()`.

### Impact

Purely a code clarity issue. The double path adds ~150 lines of complexity for no functional benefit.

### Fix (Optional)

Consolidate into a single try/except chain:

```python
def _execute_with_native_resolution(self, session_id, keyword_name, arguments, 
                                      namespace, variables, assign_to=None):
    runner = namespace.get_runner(keyword_name)
    data_kw = RunKeyword(name=keyword_name, args=tuple(arguments),
                         assign=tuple(assign_to) if assign_to else ())
    res_kw = ResultKeyword(...)
    ctx = EXECUTION_CONTEXTS.current
    
    try:
        result = runner.run(data_kw, res_kw, ctx)
    except ExecutionStatus:
        raise  # assertion failures, keyword errors — propagate
    except Exception:
        # Last resort: BuiltIn.run_keyword (handles edge cases)
        result = BuiltIn().run_keyword(keyword_name, *arguments)
    
    # Handle assignment only if RunKeyword.assign was empty
    ...
    return {"success": True, "result": result, ...}
```

---

## Finding 6 — ✅ Core Architecture Is Sound

### What rf-mcp Gets Right

1. **`Namespace` + `EXECUTION_CONTEXTS` pattern**: The fundamental approach of creating a `Namespace`, pushing an `_ExecutionContext`, and using `namespace.get_runner()` to resolve keywords is exactly how RF works internally. This is correct.

2. **`BuiltIn.run_keyword()` fallback**: Using `BuiltIn().run_keyword()` as a fallback is the safest escape hatch. It delegates to the active context's namespace and goes through RF's full argument resolution pipeline.

3. **`_suppress_stdout()` fd redirect**: The reference-counted fd 1 redirect to prevent RF console output from corrupting MCP's stdio transport is well-engineered.

4. **`console='none'` in `RobotSettings`**: This is the correct way to suppress console output at the RF settings level.

5. **Session variable sync**: Syncing session variables into RF `VariableScopes` before keyword execution (line 516-526) ensures keywords like `Evaluate` can reference them.

6. **Search order restore**: Re-applying `namespace.set_search_order()` before each execution (line 554) correctly handles the case where multiple sessions share one `EXECUTION_CONTEXTS`.

7. **`ExecutionStatus` isinstance guard**: The check at line 643 correctly short-circuits on RF execution errors, preventing duplicate StatusReporter cycles.

8. **ADR-005 multi-test cycling**: The concept of `start_test_in_context` / `end_test_in_context` for proper test isolation is architecturally correct. The implementation just needs the double-call fix (Finding 1).

---

## Summary of Recommended Changes

| # | Severity | Finding | Fix Effort |
|---|----------|---------|------------|
| 1 | 🔴 BUG | Double `start_test`/`end_test` → zombie scope + TEST-scope lib corruption | Remove 3 lines |
| 2 | 🟡 ISSUE | Lifecycle ordering differs from RF native | Reorder ~15 lines in `create_context_for_session` |
| 3 | 🟡 ISSUE | Direct method bypass skips variable resolution | Delete `_try_execute_from_library` (~30 lines) |
| 4 | 🟢 MINOR | Double variable assignment | 2-line guard (optional) |
| 5 | 🟢 MINOR | Double execution path (generic → outer) | Consolidate (optional, reduces ~150 lines) |

### Priority Fix — Minimal Diff

The most impactful fix is removing the three redundant calls (Finding 1). In `create_context_for_session()`:

```diff
- namespace.start_suite()
- namespace.start_test()
- # Also start a real running+result test in ExecutionContext
  try:
      run_test = RunTest(name=f"MCP_Test_{session_id}")
      res_test = ResTest(name=f"MCP_Test_{session_id}")
+     namespace.start_suite()
      ctx.start_test(run_test, res_test)
```

In `start_test_in_context()`:

```diff
- namespace.start_test()
  ctx.start_test(run_test, res_test)
```

In `end_test_in_context()`:

```diff
  ctx.end_test(res_test)
- namespace.end_test()
```

---

## Appendix: Experiment Results

All experiments were run on RF 7.4.2 and are fully reproducible.

| Experiment | What It Tested | Key Finding |
|---|---|---|
| 1 — Ordering | rf-mcp vs RF-native lifecycle order | `import_library("BuiltIn")` before context push fails with `AttributeError` |
| 2 — Double Exec | runner.run() + fallback interaction | No actual double execution if generic path succeeds |
| 3 — Assign | runner.run(assign=) behavior | RF auto-assigns; rf-mcp's manual assign is redundant |
| 4 — Bypass | Direct method call behavior | `${variables}` are NOT resolved in direct calls |
| 5 — Scopes | Double start_test scope accumulation | 4 scopes instead of 3; balanced by double end_test |
| 6 — ADR-005 | Multi-test cycling with double calls | Push/pop balanced but wasteful; TEST-scope libs at risk |
| 7 — Zombie | Variable behavior with zombie scope | Set Test Variable works; zombie scope is benign for standard libs |
| 8 — TEST-scope | Double end_test on TEST-scope libraries | Would corrupt instance_cache; safe for GLOBAL/SUITE libs |
