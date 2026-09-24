# Proposal: quality-baseline-cleanup

## Why

The `Quality` job added by `ci-quality-scanning` reported 439 findings on its first run.
Reading them found **three latent `NameError` bugs in shipped code** - not style issues,
crashes waiting for the right code path:

**1. An error handler that crashes instead of reporting**
`src/robotmcp/components/execution/suite_execution_service.py:28`

```python
except ImportError:
    ROBOT_AVAILABLE = False
    logger.warning("Robot Framework not available - suite execution will be limited")
...
logger = logging.getLogger(__name__)      # defined FOUR LINES LATER
```

If Robot Framework is ever unimportable, the handler raises
`NameError: name 'logger' is not defined` instead of logging the warning it was written
to log - turning a degraded-mode message into an import-time crash.

**2. A name used inside the expression that creates it**
`src/robotmcp/domains/snapshot/aggregates.py:150`

```python
new_snapshot = Snapshot(
    ...
    token_estimate_after=new_snapshot._estimate_tokens_for_tree(new_tree)   # not bound yet
)
```

`new_snapshot` is referenced while still being constructed, so this raises `NameError`
whenever the compression path runs.

**3. A nested helper called from another scope**
`src/robotmcp/components/execution/rf_native_context_manager.py:975` calls
`_normalize_arg`, which is defined at line 596 as a nested function inside a *different*
function. It is not in scope at the call site, so the BuiltIn fallback path raises
`NameError` - on the fallback, which is where things are already going wrong.

All three are `F821 undefined-name`. None would be caught by the test suite unless the
specific path executes, which is why 7,300 passing tests never surfaced them.

Alongside these, the baseline carries mechanical debt worth clearing while attention is
on it: **132 unused imports**, **25 unused variables**, **13 placeholder f-strings** and
**21 semicolon-joined statements** - all auto-fixable, all pure noise that makes the real
findings harder to see.

## What Changes

- **Fix the three `NameError` bugs**, each with a regression test that executes the
  offending path - a fix without a test would leave them free to come back.
- **Resolve the remaining `F821` findings**, which are type annotations referencing names
  that are never imported (`Any`, `FrontendConfig`, `ExecutionSession`, `TimeoutPolicy`).
  These are harmless while annotations stay unevaluated, but they are indistinguishable
  from the real bugs in the report - clearing them is what makes `F821` a signal.
- **Apply the safe auto-fixes** (`F401`, `F841`, `F541`, `E702`) so the remaining count
  reflects decisions rather than debris.
- **Record the new baseline** so the next reading of the report starts from a known number.

## Non-Goals

- **The `S` (flake8-bandit) findings** - 160 `try-except-pass`, 22 `try-except-continue`,
  14 hardcoded SQL, 10 unguarded `subprocess`, 6 weak MD5, 2 hardcoded passwords. Each
  needs a judgement about intent, and several are deliberate (rf-mcp swallows exceptions
  on best-effort paths by design). Triaging them is real work and bundling it here would
  make this change unreviewable.
- **Widening the ruff rule set.** The selection stays as pinned; this change drives the
  existing baseline down rather than moving the goalposts.
- **The dependency advisory backlog.** Dependabot is opening those PRs; six are already
  merged. Separate track, separate risk profile.
