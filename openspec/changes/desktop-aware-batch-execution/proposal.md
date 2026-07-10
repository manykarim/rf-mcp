# Proposal: desktop-aware-batch-execution

## Why

SPIKE 2 (`experiments/SPIKE_2_desktop_efficiency.md`, 2026-07-10) measured
desktop-automation turn economy and found `execute_batch` — the single biggest
round-trip reducer rf-mcp has — is effectively unreachable and unsafe for
desktop sessions (spike recommendation #3, plus the retry-cap half of #8):

- **The win is proven.** The pre-stated `cc-desk-batch` experiment
  (`experiments/cc-desk-batch/`, spike §3.1) ran the same gnome-calculator task
  as the baseline but mandated one `execute_batch`: **70 → 11 tool calls
  (−84%, 6.4×), 410 s → 160 s (−61%), $4.98 → $0.85 (−83%)**, same PASS
  quality. PlatynUI sequences batch fine — the batch path has **no desktop
  exclusion** (`server.py:5093` ff.; steps run through the same executor via
  `batch_execution/services.py` `BatchRunner.execute`).
- **But the agents that most need it cannot see it.** The `desktop_exec` tool
  profile (`tool_profile/aggregates.py:288-306`) exposes 6 tools and omits
  `execute_batch`; profile activation removes unlisted tools from the visible
  set (`ToolProfileManager.activate_profile`, `tool_profile/services.py:112`).
  So exactly the small-context desktop models the profile targets cannot
  batch. `slim_exec` (`aggregates.py:309-327`) already ships `execute_batch`
  to 7B models — precedent that batch belongs in constrained profiles.
  Result: **0 `execute_batch` uses across all 4 spike transcripts** (§2.7).
- **Batch recovery is browser-only and actively harmful on desktop.** The
  default strategy catalog (`domains/recovery/aggregates.py:218-336`) is
  `Execute Javascript` scroll / DOM-overlay removal, `Handle Alert`,
  `Go Back`, `Reload Page` — all meaningless in a PlatynUI session. Worse,
  each blind retry of a failed desktop step re-pays PlatynUI's full
  descriptor-resolution budget (`QuerySettings(30, 0.1)` on the `BareMetal`
  library; desktop sessions intentionally skip rf-mcp timeout injection,
  `keyword_executor.py:2843-2852` "Never inject"). Spike §3.2: one bad step
  burned **93,237 ms — 78% of the 120,000 ms default batch budget**
  (`BatchTimeout.DEFAULT_MS`, `batch_execution/value_objects.py:147`).
- **Code-level experiment (this proposal — reproduces the 93 s without a
  container).** Running the real classifier
  (`RecoveryEngine.with_defaults().classify(...)`) on PlatynUI's actual error
  text — `ElementNotFoundError: No UiNode found for UiNodeDescriptor query
  '…' within timeout of 30 seconds.` — classifies it as **`TimeoutException`,
  not `ElementNotFound`**: the "timeout of 30 seconds" wording matches the
  generic timeout pattern (priority 8, `recovery/aggregates.py:156-160`),
  while no ELEMENT_NOT_FOUND alternative matches "No UiNode found"
  (`:140-145` — it requires literal "not found", "no element", etc.). With the
  `on_failure="recover"` default the selected strategies are: attempt 1
  `extended_timeout` (empty actions → immediate retry → **+30 s**), attempt 2
  `reload_page` (`Reload Page` — no such keyword in a desktop session — plus
  `Sleep 3s` → retry → **+30 s**). Arithmetic: 30 s initial + 30 s + 3 s +
  30 s ≈ **93 s**, matching the observed 93,237 ms. Recovery *multiplies* the
  hang instead of containing it — confirming the spike's assumption that the
  browser-only tiers are exercised wastefully on desktop.
- **Nothing steers desktop agents toward batch** (§2.7): no instruction,
  hint, or init guidance mentions `execute_batch` in a desktop context.

**Relationship to spike #8:** its argument-list-validation half (dict
`arguments` silently resolving to the dict's KEYS — the trigger of the §3.2
burn) is **already shipped** in the `agent-ergonomics-fixes` change
(`BatchExecution.create` / `_resolve_step_args` now reject malformed steps
with actionable errors, `batch_execution/aggregates.py:107-120`). This
proposal covers only the *remaining* half of #8 — the desktop retry timeout
cap — plus all of #3.

## What Changes

- **Profile inclusion** — add `execute_batch` to the `desktop_exec` preset
  (`tool_profile/aggregates.py:294-300`, 6 → 7 tools; update the token
  estimate in the docstring). This matches `browser_exec`'s 7-tool size and
  follows the `slim_exec` precedent of shipping batch to constrained models.
- **Desktop error classification** — register a higher-priority error pattern
  (e.g. priority 12, above the generic timeout pattern's 8) matching PlatynUI
  element-resolution failures (`No UiNode found` / `ElementNotFoundError`) so
  they classify as `ELEMENT_NOT_FOUND` instead of `TimeoutException`.
- **Desktop recovery tiers** — platform-aware strategy selection: a
  `platform` dimension on `RecoveryStrategy` selection (see `design.md` for
  placement) so desktop sessions get a desktop catalog — Tier 1
  `desktop_wait_and_retry` (`Sleep` then retry), Tier 2
  `desktop_activate_window` (re-`Activate Window` on the current root, then
  retry) and re-query of the scoped root/`control:Frame` — and never execute
  the browser-only actions (`Execute Javascript`, `Reload Page`, `Go Back`,
  `Handle Alert`). Web sessions keep the existing catalog unchanged. The
  adapter resolves platform via `session.is_desktop_session()`
  (`models/session_models.py:274`).
- **Descriptor-resolution timeout cap during batch recovery retries** — on
  desktop, recovery *retries* of a failed step run with the `BareMetal`
  `query_settings.timeout` temporarily capped (default 5 s, restore in
  `finally`); the *initial* attempt keeps the native 30 s budget. Mechanism:
  the live library instance is reachable via
  `namespace.get_library_instance("PlatynUI.BareMetal")` (same pattern as
  `session_manager.py:132`), and `QuerySettings` is a plain mutable dataclass.
  Worst case for one failed desktop step drops from ~93 s to ~45 s (30 s
  initial + Sleep + 2 × capped retries), keeping the 120 s batch budget
  survivable.
- **Desktop retry safety (spray-click guard)** — in desktop sessions,
  recovery retries are permitted only for error classes where the input
  provably never fired (`ELEMENT_NOT_FOUND`: descriptor resolution precedes
  the pointer/keyboard action). Post-action failures behave as
  `on_failure="stop"` regardless of policy, so a batch cannot blind-repeat
  clicks or keystrokes against an unknown desktop state. `design.md`
  discusses the `on_failure` default tradeoff.
- **Batch-first init steering** — the desktop session init guidance gains a
  one-line pointer to `execute_batch` for multi-step interaction sequences.
  The desktop init cheat-sheet itself (keyword surface, locator crib) is
  owned by the sibling `desktop-turn-economy-guidance` change (spike §5 item
  1); this change only adds the batch-steering sentence and does not
  duplicate that content.

Out of scope: batch argument validation (shipped in `agent-ergonomics-fixes`);
`browser_exec`/`api_exec` batch inclusion (no spike evidence yet); desktop
failure-evidence collection (`EvidenceCollectorImpl` uses Browser keywords —
they fail fast and soft on desktop; noted as follow-up in `design.md`).

## Capabilities

### New Capabilities

- `desktop-aware-batch-execution`: desktop sessions can and should use
  `execute_batch` — the tool is visible in the `desktop_exec` profile,
  PlatynUI failures classify correctly, recovery uses desktop-appropriate
  actions with a capped descriptor-resolution timeout on retries, blind
  retries are restricted to provably-unfired inputs, and desktop init
  guidance steers agents toward batching.

### Modified Capabilities

- None (there is no existing recovery-strategy or tool-profile capability
  spec under `openspec/specs/`; all requirements are additive).

## Impact

- `src/robotmcp/domains/tool_profile/aggregates.py:294-300` — `desktop_exec`
  preset gains `execute_batch`.
- `src/robotmcp/domains/recovery/aggregates.py` — new PlatynUI error pattern
  (`_register_default_patterns`, `:131-187`); desktop strategy registrations
  (`_register_default_strategies`, `:193-344`); platform-aware
  `select_strategy` (`:61-100`).
- `src/robotmcp/domains/recovery/value_objects.py:91-119` — platform
  dimension on `RecoveryStrategy`.
- `src/robotmcp/adapters/recovery_adapter.py:43-80` — resolve session
  platform, pass it to strategy selection.
- `src/robotmcp/domains/batch_execution/services.py:186-266`
  (`BatchRunner._handle_failure`) — desktop retry gate (unfired-only) and
  capped-timeout retry hook.
- `src/robotmcp/components/execution/keyword_executor.py` (or a small helper
  beside it) — the query-timeout cap context manager using
  `get_library_instance("PlatynUI.BareMetal")`.
- Desktop init guidance text (same surface the sibling
  `desktop-turn-economy-guidance` change extends) — one batch-steering line.
- Tests: `tests/unit/test_desktop_aware_batch_execution.py` — profile
  contains the tool; classification of the real PlatynUI error string;
  desktop vs web strategy selection (no browser-only actions on desktop; web
  catalog unchanged); timeout cap applied on retry and restored on exit
  (including exception paths); unfired-only retry gate; guidance text
  assertion. Existing batch/recovery suites
  (`tests/unit/domains/batch_execution/`, `tests/unit/test_intent_registry.py`
  neighbors) must stay green.
