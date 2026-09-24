# Tasks: quality-baseline-cleanup

Bugs first, sweep second - so the three diffs that matter are not buried in a mechanical
changeset.

## 1. Reproduce before fixing
- [x] 1.1 Regression test for `suite_execution_service.py:28` that executes the ImportError handler (simulate Robot Framework being unimportable) and asserts it warns rather than raising NameError
- [x] 1.2 Regression test for `aggregates.py:150` that runs the snapshot compression path and asserts it completes
- [x] 1.3 Regression test for `rf_native_context_manager.py:975` that reaches the BuiltIn fallback
- [x] 1.4 Confirm all three FAIL against the current code - a regression test that passes before the fix proves nothing

## 2. Fix the three NameErrors
- [x] 2.1 `suite_execution_service.py`: move `logger = logging.getLogger(__name__)` above the try/except that uses it
- [x] 2.2 `aggregates.py`: compute the token estimate without referencing the not-yet-bound `new_snapshot`, PRESERVING the value rather than dropping the field
- [x] 2.3 `rf_native_context_manager.py`: make `_normalize_arg` reachable from line 975 without breaking its use at line 596
- [x] 2.4 Confirm the three tests from section 1 now pass

### 2.5 Extra defects uncovered by the section-1 tests (not F821-visible)
The `aggregates.py` regression test was the first thing ever to execute
`PageSnapshot.fold_lists` - it has no callers in `src/` or `tests/` (the passing
`test_snapshot_domain.py` tests exercise a *different* `PageSnapshot` defined inside
that test file). Past the NameError sat three AttributeErrors that ruff cannot see:
- [x] 2.5.1 `_create_folded_list`: `ElementRef.index` does not exist (it has `.value` / `to_index()`), and the f-string also double-prefixed the "e"
- [x] 2.5.2 `fold_lists`: `AriaTree._recalculate_counts()` does not exist - `node_count`/`interactive_count` are computed properties; the two sibling calls in `services.py` are `hasattr`-guarded, this one was not
- [x] 2.5.3 `_estimate_tokens_for_tree` + `find_element_by_role`: `traverse()` is on `AriaNode`, not on the `entities.AriaTree` this module imports - walk `tree.root.traverse()`. NOTE: `estimate_tokens()` is public and has live callers, so this one was reachable outside the dead fold path
- [x] 2.5.4 Regression test `test_folded_summary_reports_a_usable_ref_range` locks in the ref-range shape (a naive `.to_index` swap would have produced "ee3")

## 3. Clear the annotation-only F821s
- [x] 3.1 `plugins/manager.py`: import `Any` from typing (9 findings)
- [x] 3.2 `server.py`: import `FrontendConfig` and `ExecutionSession`, under TYPE_CHECKING if the runtime import would be circular or costly
- [x] 3.3 `keyword_classifier.py`: import `TimeoutPolicy`
- [x] 3.4 `ruff check src/ --select F821` reports zero

## 4. Safe auto-fix sweep
- [x] 4.1 `ruff check src/ --fix` for F401/F841/F541/E702 only - do NOT pass `--unsafe-fixes`
- [x] 4.2 Review the diff: an "unused" import may be a re-export or an import for side effects
- [x] 4.3 Full suite green with no fewer tests collected than before

## 5. Verify and record
- [x] 5.1 Record the new finding count and confirm the drop is accounted for

**439 -> 295 (-144), fully accounted, no rule went up:**

| rule | HEAD | now | delta | |
|---|---|---|---|---|
| F821 undefined-name | 17 | 0 | -17 | 3 real NameErrors + 14 annotation-only |
| F401 unused-import | 132 | 23 | -109 | sweep; 23 remain as unsafe-only fixes |
| F541 f-string-no-placeholder | 13 | 0 | -13 | sweep |
| F841 unused-variable | 25 | 21 | -4 | 21 remain as unsafe-only fixes |
| F811 redefined-while-unused | 2 | 1 | -1 | sweep |
| E402 import-not-at-top | 2 | 2 | 0 | briefly +1 from the logger hoist; resolved |
| everything else (S*, E7xx) | 222 | 222 | 0 | untouched - needs judgement, not a sweep |

The 295 remainder is dominated by S110 try-except-pass (160). None of it is safely
auto-fixable: `--fix` now reports 1 fixable, 21 unsafe-only.
- [ ] 5.2 Confirm the `Quality` job still passes and its digest reflects the lower number
- [ ] 5.3 Update the baseline recorded in the PR description
