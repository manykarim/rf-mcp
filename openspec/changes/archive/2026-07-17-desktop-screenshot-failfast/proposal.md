# Proposal: desktop-screenshot-failfast

## Why

SPIKE 2 (`experiments/SPIKE_2_desktop_efficiency.md`, 2026-07-10) measured where
desktop-automation turns go and found two reproducible 30-second-hang traps that
a pre-dispatch guard — the family that already exists for unscoped `//` locators
— would convert into instant, actionable refusals (spike recommendations #2 and
#4, both rated small effort / low risk):

- **Trap 1 — `Take Screenshot` filename-in-descriptor-slot (spike §2.3).**
  PlatynUI's signature is `take_screenshot(descriptor=None, filename=…, rect=None)`
  (verified via `inspect.signature` on `PlatynUI.BareMetal`). A filename-only
  positional (`Take Screenshot    /artifacts/calc.png`) binds the path to
  `descriptor`; the native runtime then retries descriptor resolution for its
  full 30 s budget before `ElementNotFoundError`. Desktop sessions intentionally
  skip timeout injection (`keyword_executor.py:2843-2852` — "Never inject"), so
  nothing caps the wait. In the desk-calc run this cost 4 tool calls (3 failures
  + success) and ≥35 s wall time (spike §1.3, call 30). The executor **already
  recognizes** bare image-path positionals — `screenshot_request_path()`
  (`desktop_execution_signals.py:245-264`) extracts them for the D3 evidence
  path guard — but no fail-fast "your first positional is a filename" hint exists.

- **Trap 2 — `control:Window` on Linux (spike §2.4).** On AT-SPI, top-level
  windows have role Frame; `control:Window` matches only compositor/shell
  elements, so a `control:Window` descriptor burns the full 30 s retry before
  `ElementNotFoundError` (cc-desk-base run, spike §1.2 failure list). The locator
  guidance documents this exact fact (`rf_native_type_converter.py:1727-1731`,
  "Window on Linux (Frame!)" example at `:1713`) — but guidance is opt-in and
  the cc-desk-base agent never called `get_locator_guidance` (spike §1.2);
  nothing intercepts the doomed query.

The pattern to mirror is proven: the unscoped-`//` refusal guard
(`_unscoped_locator_guard`, `keyword_executor.py:626-704`, pure detection in
`desktop_execution_signals.py` `is_query_keyword`/`is_unscoped_locator`
`:77-108`) fired correctly in **both** spike runs, converting what used to be a
transport-killing multi-second walk into a 0 s hint with a concrete rewrite and
an explicit opt-out (`ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED` / session flag,
downgrading refusal to a one-time warning).

**Code-level experiment (this proposal):** calling `screenshot_request_path()`
directly confirms it returns the image path for *both* the trap shape
(`["/artifacts/x.png"]` → `/artifacts/x.png`) and the correct shape
(`[descriptor, "/artifacts/x.png"]` → `/artifacts/x.png`), and `None` for
descriptor-only / `EMBED` args. It answers "is there a path-looking arg?" but
not "*which slot* is it in?" — so the new guard needs a sibling pure predicate
that is positional-slot-aware, sharing the extension list and `name=value`
parsing already in `desktop_execution_signals.py:238-264`.

## What Changes

- **Screenshot-signature guard (#2)** — a new pure predicate in
  `desktop_execution_signals.py` (beside `screenshot_request_path`, reusing
  `_SCREENSHOT_EXTENSIONS` and the `name=value` parsing) detects a desktop
  `Take Screenshot` whose **descriptor slot** (first positional, or explicit
  `descriptor=`) is a bare image path. A new executor guard method refuses the
  step pre-dispatch with a structured failure that states the
  `(descriptor, filename, rect)` signature and both correct forms
  (`Take Screenshot    filename=/path/x.png` for the whole desktop, or
  `Take Screenshot    <descriptor>    /path/x.png` for an element). Named
  `filename=`-only calls and correct two-positional calls proceed unchanged.

- **Linux `control:Window` guard (#4)** — a new pure predicate detects a
  `control:Window` role token in the descriptor/XPath argument of a desktop
  tree-resolving keyword (`Query`/`Evaluate`/`Set Root`/`Get Attribute` and the
  pointer/keyboard interaction keywords — the existing
  `_TREE_RESOLVING_KEYWORDS`/`_XPATH_KEYWORDS` frozensets at
  `desktop_execution_signals.py:39-42,69`, extended to cover `Evaluate`). On
  Linux only (`sys.platform == "linux"`, same check as
  `platynui_plugin.py:334`), the executor refuses pre-dispatch with a hint that
  states the AT-SPI Frame-vs-Window fact and offers the concrete rewrite
  (`control:Window` → `control:Frame` in the submitted locator). We refuse with
  a rewrite *hint* rather than silently rewriting, because a silent mutation
  would misrepresent what executed and diverge from the step recorded for
  `build_test_suite` — consistent with the unscoped guard's refuse-and-hint
  contract.

- **Opt-outs consistent with the guard family** — each guard honors an explicit
  escape hatch mirroring `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED`
  (`keyword_executor.py:658-682`): `ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR`
  and `ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW` env vars (or the session-flag
  equivalents), which downgrade the refusal to a one-time warning and let the
  step proceed — for the rare deliberate cases (an element genuinely named like
  a path; a compositor/shell `control:Window` query on Linux).

- **Wiring** — both guards run in the same desktop-only pre-dispatch block as
  the existing guards (`keyword_executor.py:1822-1836`, between the D3
  screenshot-path guard and the unscoped-locator guard), so they apply to
  `execute_step`, `intent_action`, **and `execute_batch` steps** (batch steps
  run through the same executor — spike §2.7), and are inert for web/API
  sessions (Browser's `Take Screenshot` with a path positional stays valid).

Out of scope (sibling spike changes): init-time keyword cheat-sheet + Process
core (`desktop-turn-economy-guidance`, levers #1/#5), desktop-aware batch
recovery + argument validation (`desktop-aware-batch-execution`, levers #3/#8),
actionable-controls view (lever #6).

## Capabilities

### New Capabilities

- `desktop-screenshot-failfast`: a desktop `Take Screenshot` whose descriptor
  slot holds a bare image path, and a Linux desktop locator using
  `control:Window`, are refused before native dispatch with an actionable
  signature/rewrite hint instead of burning the 30 s descriptor-resolution
  retry; each guard has an explicit opt-out that downgrades the refusal to a
  one-time warning, mirroring the unscoped-locator guard.

## Impact

- `src/robotmcp/components/execution/desktop_execution_signals.py` — two new
  pure predicates beside `screenshot_request_path()` (:245-264): a
  descriptor-slot path-misuse detector (shares `_SCREENSHOT_EXTENSIONS` :238
  and the `name=value` parsing) and a `control:Window` locator detector
  (reuses/extends `_TREE_RESOLVING_KEYWORDS` :39-42 and `_XPATH_KEYWORDS` :69).
- `src/robotmcp/components/execution/keyword_executor.py` — two new guard
  methods modeled on `_unscoped_locator_guard` (:626-704), wired into the
  desktop pre-dispatch block (:1822-1836); opt-out handling mirrors :658-682.
- Tests: `tests/unit/test_desktop_screenshot_failfast.py` — pure-predicate
  cases (trap shape, correct shapes, `EMBED`, named forms), guard refusal
  payloads (signature hint / Frame rewrite), platform gating (Linux-only for
  `control:Window`), opt-out downgrade to one-time warning, non-desktop
  sessions untouched — mirroring `tests/unit/test_unscoped_locator_guardrail.py`.
- No behavior change for web/API sessions, for correct desktop calls, or for
  Windows (`control:Window` is correct under UIA — `rf_native_type_converter.py:1732`).
