# Tasks: desktop-screenshot-failfast

## 1. Pure detection predicates (desktop_execution_signals.py)
- [ ] 1.1 Add a descriptor-slot path-misuse predicate beside
      `screenshot_request_path()` (:245-264): returns the offending path when a
      `Take Screenshot` **first positional** (or explicit `descriptor=`) is a
      bare image path (reuse `_SCREENSHOT_EXTENSIONS` and the `name=value`
      parsing); returns None for named `filename=`-only calls, correct
      `(descriptor, path)` two-positional calls, `EMBED`, and templated names.
      Pure — no I/O, unit-testable without a desktop.
- [ ] 1.2 Add a `control:Window` locator predicate: returns the offending
      locator when the descriptor/XPath argument of a desktop tree-resolving
      keyword contains the `control:Window` role token. Reuse/extend
      `_TREE_RESOLVING_KEYWORDS` (:39-42) and `_XPATH_KEYWORDS` (:69) so
      `Query`, `Evaluate`, `Set Root`, `Get Attribute`, and the
      pointer/keyboard interaction keywords are all covered. Platform gating
      stays OUT of the pure function (caller supplies it).

## 2. Executor guards (keyword_executor.py)
- [ ] 2.1 `_screenshot_signature_guard`: modeled on `_unscoped_locator_guard`
      (:626-704). On detection, return a structured `success:false` error whose
      hint states the `(descriptor, filename, rect)` signature and both correct
      forms (`filename=/path/x.png` for whole desktop; `<descriptor>  /path/x.png`
      for an element). Honor `ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR` env var /
      `platynui_allow_path_descriptor` session flag → proceed with a ONE-TIME
      warning (mirror :658-682).
- [ ] 2.2 `_control_window_guard`: Linux-only (`sys.platform == "linux"`, as
      `platynui_plugin.py:334`). On detection, refuse with a hint stating the
      AT-SPI Frame-vs-Window fact (`rf_native_type_converter.py:1727-1731`) and
      the concrete rewrite of the submitted locator
      (`control:Window` → `control:Frame`). Honor
      `ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW` env var /
      `platynui_allow_control_window` session flag → proceed with a ONE-TIME
      warning. Do NOT auto-rewrite (recorded steps must match what executed).
- [ ] 2.3 Wire both guards into the desktop-only pre-dispatch block
      (:1822-1836), adjacent to `_screenshot_path_guard` and
      `_unscoped_locator_guard`, so they cover execute_step, intent_action, and
      execute_batch steps, and are inert for non-desktop sessions.

## 3. Tests + validation
- [ ] 3.1 `tests/unit/test_desktop_screenshot_failfast.py`, mirroring
      `tests/unit/test_unscoped_locator_guardrail.py`:
      (a) predicate truth table — trap shape `["/artifacts/x.png"]` detected;
      `[descriptor, "/x.png"]`, `["filename=/x.png"]`, `EMBED`, `{index}`, and
      non-screenshot keywords not detected;
      (b) `control:Window` detected in Query/Evaluate/Set Root/Pointer Click
      descriptors; `control:Frame` and Windows-platform calls untouched;
      (c) guard payloads — structured failure (not exception), hint text names
      the signature / the Frame rewrite of the exact submitted locator;
      (d) opt-outs — env var and session flag each downgrade to a one-time
      warning and the step proceeds; warning appears at most once per session;
      (e) non-desktop session → both guards return None (Browser
      `Take Screenshot  /path.png` unaffected).
- [ ] 3.2 Full unit suite green (`uv run pytest tests/unit/`) — no regressions;
      confirm the unscoped-locator guardrail tests still pass unchanged.
