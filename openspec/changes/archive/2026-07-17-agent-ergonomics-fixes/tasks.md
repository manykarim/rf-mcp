# Tasks: agent-ergonomics-fixes

## 1. F2 — actionable execute_batch step validation
- [x] 1.1 `BatchExecution.create` (`aggregates.py`): before building each `BatchStep`, validate `s` is a dict with a non-empty string `keyword`; raise `ValueError("Step {i}: …")` naming the required shape. Keep the existing `_resolve_step_args` arg handling.
- [x] 1.2 Ensure the server `execute_batch` surfaces the ValueError as a structured `success:false` error (not a raw exception).
- [x] 1.3 `_resolve_step_args`: reject non-list `arguments`/`args` (dict/str) with a step-indexed "must be a list" error — `list(dict)` used to silently yield the dict's KEYS (spike §3.2: a dict arg burned 93s in desktop batch retries). Parity with execute_step.

## 2. F5 — allow standard utility libraries in every session type
- [x] 2.1 `_get_allowed_libraries_for_session_type` (`session_models.py`): always add `OperatingSystem, Collections, String, DateTime, Process` (with the existing `BuiltIn`). Web libs (`Browser`/`Selenium`/`Appium`) stay profile-governed.

## 3. F4 — document the execute_batch BDD limitation
- [x] 3.1 `execute_batch` docstring (`server.py`): note that batch steps do NOT support `bdd_group`/`bdd_intent`; use per-step `execute_step` for BDD grouping.

## 4. Docs — suite persistence in the WORKFLOW GUIDE
- [x] 4.1 `instruction/value_objects.py` (+ `templates/detailed.txt` if it mirrors): add a short "persist suites via `build_test_suite(output_path=…)`, never write `rf_text` via `Create File`" note.

## 5. Tests + validation
- [x] 5.1 `tests/unit/test_agent_ergonomics_fixes.py`: (a) batch with a step missing `keyword` → actionable ValueError/structured error naming the field, not `KeyError`; a valid batch still builds; (b) an `api_testing` session allows `OperatingSystem` (and the other utility libs), while `Browser` stays excluded for non-web types; (c) the WORKFLOW GUIDE text mentions `output_path`; the `execute_batch` docstring mentions the bdd_group limitation.
- [x] 5.2 Full unit suite green (no regressions).
