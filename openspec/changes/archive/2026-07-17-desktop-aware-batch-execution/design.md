# Design: desktop-aware-batch-execution

## Context

`execute_batch` (ADR-011) runs steps through the same executor as
`execute_step` (`server.py:5093` ff. → `BatchRunner.execute`,
`batch_execution/services.py:122-184`), so desktop/PlatynUI steps already
work — the `cc-desk-batch` experiment proved it (70 → 11 calls). What is
desktop-*blind* is everything around the happy path:

1. the `desktop_exec` profile hides the tool;
2. the recovery catalog (`recovery/aggregates.py:218-336`) assumes a browser;
3. every retry of a failed PlatynUI step re-pays the native 30 s
   descriptor-resolution budget (`QuerySettings(30, 0.1, False)` set in
   `BareMetal.__init__`; rf-mcp deliberately never injects timeouts for
   desktop, `keyword_executor.py:2843-2852`).

The measured failure mode (spike §3.2, reproduced code-level in the
proposal): PlatynUI's `ElementNotFoundError` text contains "timeout of 30
seconds", so it classifies as `TimeoutException` → `extended_timeout`
(attempt 1, bare retry) → `reload_page` (attempt 2) → 30 + 30 + 3 + 30 ≈ 93 s
for one step against a 120 s budget.

## Goals / Non-Goals

**Goals**
- Make `execute_batch` visible to desktop-profiled agents.
- Make a failed desktop batch step cost bounded and its recovery actions
  meaningful.
- Never let batch recovery blind-repeat an input that may already have fired.

**Non-Goals**
- Redesigning the recovery domain (the tiered engine stays; it gains a
  platform dimension).
- Desktop failure evidence (`EvidenceCollectorImpl` calls Browser
  `Screenshot`/`Get Source` — on desktop these fail fast on keyword lookup
  and are soft-logged; a `Take Screenshot`-based desktop evidence path is a
  possible follow-up, not required for turn economy).
- Batch in `browser_exec`/`api_exec` (no spike evidence of need; those
  domains have cheap failures).

## Decision 1 — Where platform awareness lives

**Chosen: platform as a first-class selection dimension in the recovery
domain**, resolved once per recovery attempt by the adapter.

- `RecoveryStrategy` (`recovery/value_objects.py:91-119`) gains
  `platforms: frozenset[str]` defaulting to `{"web"}` so all nine existing
  strategies keep their semantics without edits at call sites that construct
  them (the default preserves current behavior for the existing catalog;
  desktop strategies register with `{"desktop"}`; a strategy valid everywhere
  may use `{"web", "desktop"}` — e.g. plain Sleep-based waits).
- `RecoveryEngine.select_strategy(classification, attempt_number,
  platform="web")` filters by `platform in strategy.platforms` in both the
  preferred-tier pass and the fallback pass (`recovery/aggregates.py:86-100`).
  Default `"web"` keeps every existing caller and test unchanged.
- `RecoveryServiceAdapter.attempt_recovery` (`adapters/recovery_adapter.py:43`)
  resolves the platform from the session:
  `execution_engine.session_manager.get_session(session_id)` →
  `session.is_desktop_session()` (`session_models.py:274`) → `"desktop"` else
  `"web"`. The adapter already receives `session_id`; it needs a reference to
  the session manager (constructor injection at the `_get_batch_runner`
  wiring point, `server.py:5057-5089`).

**Alternative considered — a second, desktop-only `RecoveryEngine` in the
container**: rejected. Classification patterns must be shared (a desktop
session can still hit generic timeout text), and two engines would duplicate
the pattern registry and drift.

## Decision 2 — Fixing classification before adding strategies

Adding desktop strategies without fixing classification would be useless: the
PlatynUI error classifies as `TIMEOUT_EXCEPTION` today, so an
`ELEMENT_NOT_FOUND`-keyed desktop strategy would never be selected.

New pattern, priority 12 (above ELEMENT_NOT_FOUND's generic 10 and — the
critical part — above TIMEOUT's 8, below the Variable guard's 15):

```
ELEMENT_NOT_FOUND ← r"No UiNode found|ElementNotFoundError|UiNodeDescriptor"
```

This is desktop-vocabulary-specific, so it cannot misfire on browser errors.
Web classification is untouched (the pattern's vocabulary never appears in
Browser/Selenium messages).

## Decision 3 — Timeout cap mechanism and scope

**Chosen: temporarily mutate `BareMetal.query_settings.timeout` around
recovery retries only, restore in `finally`.**

- `QuerySettings` is a plain mutable dataclass on the live library instance;
  the instance is reachable via
  `namespace.get_library_instance("PlatynUI.BareMetal")` — the exact pattern
  already used for RequestsLibrary (`session_manager.py:132`,
  `library_manager.py:445`). No PlatynUI keyword exists to set it, so
  instance mutation is the only lever short of a PlatynUI upstream change.
- Scope: **retries only** (the loop body in `BatchRunner._handle_failure`,
  `services.py:209-252`). The initial attempt keeps the full native 30 s —
  first resolution legitimately needs time on a busy desktop (window still
  mapping, AT-SPI tree settling). If 30 s did not find the element, a second
  30 s rarely will; the retry exists to catch *transient* states, which a
  5 s capped resolve after a `Sleep` covers.
- Default cap 5 s, env-overridable (`ROBOTMCP_BATCH_DESKTOP_RETRY_TIMEOUT`),
  matching the guard-family convention of `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED`.
- Restore MUST be in `finally`: the RF context is process-global
  (`EXECUTION_CONTEXTS.current` singleton) and `BareMetal` has SUITE scope —
  a leaked 5 s timeout would silently degrade every later stepwise
  `execute_step` in the session. Batch execution is sequential
  (`BatchRunner.execute` is a plain `for` loop), so there is no concurrent
  writer within a batch; the restore-on-exit invariant is what protects
  everything after the batch.
- Budget math with the fix (defaults: `recover`, 2 attempts): 30 s initial +
  2 s Sleep + 5 s retry + activate-window + 5 s retry ≈ **~45 s worst case**
  per failed step vs ~93 s today — two failing steps no longer guarantee a
  batch `TIMEOUT`.

**Alternative considered — cap the initial resolution too**: rejected; it
would regress legitimate slow-appearing windows and contradicts the
"never inject" decision at `keyword_executor.py:2843-2852`, which exists
because descriptor retry IS the desktop wait mechanism.

## Decision 4 — Desktop recovery catalog

| Strategy | Tier | Applies to | Actions |
|---|---|---|---|
| `desktop_wait_and_retry` | 1 | ELEMENT_NOT_FOUND | `Sleep 2s` (then capped retry) |
| `desktop_activate_window` | 2 | ELEMENT_NOT_FOUND | `Activate Window` on the session's current root descriptor, `Sleep 1s` (then capped retry) |

Deliberately small. `Activate Window` addresses the dominant real cause of
desktop element-not-found: the AUT lost focus/raise (the whole subject of
ADR-026 focused execution). Re-querying the scoped root (`control:Frame` on
Linux) happens implicitly — PlatynUI clears its runtime cache between resolve
attempts (`runtime.clear_cache()` in the descriptor loop), so the retry after
`Activate Window` re-resolves from scratch; no separate "re-Query Frame"
action is needed as a keyword, but the Tier-2 strategy must use the
*session's current root* (from `PLATYNUI_ROOT_DESCRIPTOR`), not an unscoped
query, or it would trip the unscoped-locator guard.

No desktop strategy is registered for `ELEMENT_NOT_INTERACTABLE` /
`ELEMENT_CLICK_INTERCEPTED` / `UNEXPECTED_ALERT` etc. — those classifications
are browser-vocabulary and, per Decision 5, non-resolution desktop failures
must not be retried anyway. `select_strategy` returning `None` already makes
the runner fall through to a normal failure record.

## Decision 5 — Spray-click safety and the `on_failure` default

A desktop batch failure is riskier than a web one: retried `Pointer Click` /
`Keyboard Type` steps act on whatever is focused *now* — possibly the user's
real desktop if scoping drifted (the entire concern behind ADR-027 desktop
safety). Three options considered for desktop sessions:

1. **Default `on_failure="stop"` for desktop batches.** Safest, but discards
   the recovery value entirely and silently diverges the tool's documented
   default per session type — confusing for agents reading one docstring.
2. **Keep `recover` and trust the catalog.** Unsafe: `retry` policy (no
   strategy involved) would still blind-repeat a `Keyboard Type` whose
   descriptor resolved but whose effect failed.
3. **Keep `recover` as the documented default, but gate desktop retries on
   provably-unfired inputs.** ← **Chosen.**

The gate: in a desktop session, `BatchRunner._handle_failure` enters the
retry loop only when the failure classifies as `ELEMENT_NOT_FOUND`. For
PlatynUI, descriptor resolution strictly precedes the input action
(`_resolve_screen_point` / `UiNodeDescriptor.__call__` raise before any
pointer/keyboard call), so an `ElementNotFoundError` guarantees nothing was
clicked or typed — retrying is idempotent-safe. Any other desktop failure
(action fired but errored, attribute errors, runtime errors) records a
failure immediately, exactly as `on_failure="stop"` would. This applies to
both `retry` and `recover` policies; an explicit future opt-out is possible
but not proposed (no evidence of need).

This gate is also why the change is safe to steer small models toward: the
worst a malformed desktop batch can now do is fail fast with the (already
shipped) actionable validation errors, or burn one bounded ~45 s on a
genuinely missing element.

## Decision 6 — Profile budget

`desktop_exec` grows 6 → 7 tools with COMPACT descriptions — the same size as
`browser_exec` (7 tools, ~1,750 tokens estimated) under the same
`TokenBudget.for_context_window(8192)`. `execute_batch`'s compact description
is on the order of the other execution tools (~200-300 tokens); the preset
docstring estimate updates from ~1,550 to ~1,800 tokens. If
`validate_budget` flags an overage, `activate_profile` already degrades to
MINIMAL descriptions automatically (`tool_profile/services.py:150-180`), so
no new budget machinery is needed.

## Risks

- **Leaked query-timeout mutation** (Decision 3) — mitigated by
  `finally`-restore and a dedicated test that asserts restoration on the
  exception path.
- **Pattern misfire on web errors** — the new pattern uses PlatynUI-only
  vocabulary; a regression test runs the full existing classifier test corpus.
- **`Activate Window` recovery changing focus mid-batch** — the action
  targets the session's current root descriptor only; if root is unset the
  Tier-2 strategy is skipped (strategy predicate), never falling back to an
  unscoped activation.
- **Sibling-change overlap** — the init-guidance line must land wherever
  `desktop-turn-economy-guidance` puts the desktop init payload; tasks order
  the guidance task last and mark the cross-dependency.

## Open Questions

- Should the retry cap also apply to `resume_batch` retries? (Proposed: yes —
  same code path via `BatchRunner`; called out in tasks so it isn't missed.)
- Upstream PlatynUI could expose a `Set Query Timeout` keyword, making the
  instance mutation unnecessary; worth filing upstream but not blocking.
