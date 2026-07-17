# Tasks: desktop-screenshot-failfast

> Reconciled 2026-07-17: this tasks.md was blank (0 bytes) — the apply-commit
> (aac7dd7) shipped the code but did not persist the task list. All items below
> are verified present in `src/` and covered by unit tests; restored as [x].

## 1. Screenshot-signature guard (#2)
- [x] 1.1 Pure predicate `screenshot_path_in_descriptor_slot` in `desktop_execution_signals.py` (beside `screenshot_request_path`, reusing `_SCREENSHOT_EXTENSIONS` + `name=value` parsing) — detects a `Take Screenshot` whose descriptor slot (first positional / explicit `descriptor=`) is a bare image path
- [x] 1.2 Executor guard refuses the step pre-dispatch with a structured failure naming the `(descriptor, filename, rect)` signature and both correct forms (`filename=/path` whole-desktop; `<descriptor>  /path` element)
- [x] 1.3 Named `filename=`-only and correct two-positional calls proceed unchanged; non-desktop sessions unaffected

## 2. Linux control:Window guard (#4)
- [x] 2.1 Pure predicate `control_window_locator` detects a `control:Window` role token in a tree-resolving keyword's descriptor/XPath (`_TREE_RESOLVING_KEYWORDS`/`_XPATH_KEYWORDS`, extended to cover `Evaluate`)
- [x] 2.2 Linux-only (`sys.platform == "linux"`) pre-dispatch refusal with a hint stating the AT-SPI Frame-vs-Window fact and offering the `control:Window`→`control:Frame` rewrite; locator NOT silently mutated (recorded step matches what executed)
- [x] 2.3 Non-Linux (UIA) unaffected; `control:Frame` proceeds

## 3. Opt-outs (guard-family consistent)
- [x] 3.1 `ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR` + `ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW` env vars (and session-flag equivalents), mirroring `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED`
- [x] 3.2 Opt-in downgrades the refusal to a one-time warning; refusal hint names the escape hatch

## 4. Wiring
- [x] 4.1 Both guards run in the same desktop-only pre-dispatch block in `keyword_executor.py` (between the D3 screenshot-path guard and the unscoped-locator guard) → apply to `execute_step`, `intent_action`, and `execute_batch` steps

## 5. Tests
- [x] 5.1 Unit tests for both predicates (positive/negative/opt-out/non-desktop/non-Linux) and both executor refusals — ~19 tests, green
- [x] 5.2 `openspec validate --strict` clean
