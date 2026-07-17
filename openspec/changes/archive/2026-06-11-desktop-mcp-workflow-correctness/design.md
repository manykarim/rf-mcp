## Context

Source of record: `docs/gnome-calculator-mcp-maintainer-report.md` — a full
MCP walkthrough of a GNOME Calculator stepwise scenario that never reached a
real calculator interaction because the scenario was routed through the
mobile/Appium path, cascading into 10 distinct workflow failures.

Reproduced on the current branch:
- `ExecutionSession.detect_session_type_from_scenario("Test GNOME Calculator
  desktop application…")` already returns `desktop_testing` (fixed by change
  `platynui-desktop-safety-isolation`). So finding #1's *text* path is closed.
- BUT `analyze_scenario(scenario, context="desktop")` ignores the context:
  `nlp_processor.analyze_scenario` has `context` branches for web/api/mobile/
  database only — **no `desktop`** — so an explicit desktop context does not
  force desktop routing.
- `execute_batch` parses step args at `aggregates.py` `from … steps_data` via
  `args=list(s.get("args", []))` — it reads **only** `args`, so a step passing
  `arguments` (as `execute_step` does) loses them → "expected N arguments, got
  0" (finding #8).

Overlap: findings #1 (text classification) and #3 (Process allow-list in
desktop sessions) are already addressed by `platynui-desktop-safety-isolation`.
This change targets the remaining MCP-workflow gaps (#1 context param, #2
recommender, #4 state inspection, #5 discovery, #8 batch args, #6/#7 exec env,
#9/#10 suite isolation) and composes with that work + ADR-025/026 (PlatynUI)
and ADR-027 (safety/isolation).

## Goals / Non-Goals

**Goals:**
- An explicit `context="desktop"` reliably yields a desktop/PlatynUI session.
- `recommend_libraries` prefers PlatynUI for Linux desktop scenarios.
- Desktop state inspection and keyword discovery work (no mobile-source error,
  no zero-match PlatynUI discovery).
- `execute_batch` preserves arguments under `args` or `arguments`.
- Desktop launches/recovery resolve executables consistently; Evaluate's
  expression-only limit is documented with an alternative.
- A desktop suite is generated from real interactions, not placeholders, with
  pre-start exploratory steps isolated.
- A live e2e drives the report's scenario end-to-end under the isolation
  bootstrap.

**Non-Goals:**
- Re-deriving the desktop classification / Process allow-list / safety guard
  already shipped by `platynui-desktop-safety-isolation`.
- Making `BuiltIn.Evaluate` accept statements (that is RF semantics) — we
  document and route around it, not change RF.
- New web/mobile/API behavior when the caller does not request desktop.

## Cross-LLM review (folded in)

Reviewed by Codex CLI and OpenCode (MiniMax-M3); both REQUEST_CHANGES,
source-grounded and convergent. Verified claims and consequent changes:
- **PlatynUI is NOT in `library_registry.py`** (only a runtime plugin) → new
  D0; it is the prerequisite for recommendation (D2) and discovery (D4).
- **Context routing has THREE sites**, not one: `nlp_processor.analyze_scenario`
  (no desktop branch), `detect_platform_from_scenario` (session_manager.py:287,
  mobile keywords include "app"), `configure_from_scenario` (session_models.py,
  no context param). Confirmed by the pre-existing analysis
  `docs/issues/gnome_calculator_fix_plan.md` (Fixes 1/3/4). D1 covers all three.
- **find_keywords `catalog` is a literal substring filter** (server.py) →
  the report's natural-language query returns 0 even if PlatynUI is indexed.
  D4 names this instead of "reproduce then fix".
- **Batch precedence reversed** to `arguments`-first; conflicting dual-spec is
  a validation error (D5).
- **Exec-env uses a resolution hook** (`shutil.which` on the server PATH), not
  shell inheritance (D6).
- **D7 opt-in made concrete**: `include_pre_start=False` default + payload count.

## Decisions

### D0: Register PlatynUI as a recommendable library
Add a `PlatynUI.BareMetal` entry to the library registry/recommender used by
`recommend_libraries` (categories desktop/gui, desktop use-cases), and accept
the `PlatynUI` → `PlatynUI.BareMetal` alias in discovery/recommendation.
Rationale: verified absent today; both the recommendation (D2) and discovery
(D4) fixes depend on it. Without D0 the recommender cannot rank PlatynUI at all.

### D1: Force desktop routing at ALL session-creation sites
`context="desktop"` must override routing at every site `analyze_scenario`
drives: (a) `nlp_processor.analyze_scenario` (add a `desktop` branch — none
exists), (b) `detect_platform_from_scenario` (a desktop override so the generic
"app" mobile keyword does not win), (c) `configure_from_scenario` (thread the
context so auto-configuration picks the desktop profile + Process allow-list).
Acceptance is on the **full `analyze_scenario` tool result** (session_type,
imported libraries incl. Process, search order), not just NLP output. Mirrors
the precedence in `docs/issues/gnome_calculator_fix_plan.md`. Rationale: the
caller did the right thing and was ignored at multiple layers; one layer is not
enough. Alternative considered: text-classifier only — rejected; explicit
context must win and must reach session config.

### D2: PlatynUI-first recommendation for desktop scenarios
Building on D0, rank `PlatynUI.BareMetal` ahead of `AppiumLibrary` in the
recommender for desktop context / desktop signals (the recommender already has
a `desktop` context weight and `_filter_by_context`/`_match_by_context` hooks),
and ensure the session search order places PlatynUI first. Gate strictly on
desktop context / strong desktop-only signals so mobile/web is not regressed.
Rationale: the report shows Appium led recommendations for a Linux desktop app.

### D3: Route desktop state inspection through ui_tree
In `get_session_state` page-source/state assembly, when the session is a
desktop session, use the PlatynUI `ui_tree` path (already shipped) instead of
the mobile/Appium source lookup that produced "Failed to get mobile source: No
application is open". Provide a clear desktop message when no app resolves.
Rationale: the desktop path already exists; the bug is routing — a desktop
session should never hit the mobile source path.

### D4: Named root cause for find_keywords zero-match
Two named causes (verified in source): (a) PlatynUI must be **listable by
library** — `find_keywords(library_name="PlatynUI.BareMetal")` (and the
`PlatynUI` alias) should return its keywords from the catalog; (b) the
`catalog` strategy is a **literal substring filter** on keyword/library names
(server.py), so the report's natural-language query "get window find element ui
tree" returns 0 even when PlatynUI is indexed. Fix: ensure the library listing
works (depends on D0 + the plugin's keyword surface) and either keep `catalog`
literal with a documented fallback to semantic search, or specify the catalog
query contract — agents must reach the desktop keywords. Rationale: "reproduce
then fix" was too vague; the causes are known, so the proposal lands with them.

### D5: Batch arg compatibility (`arguments`-first; reject conflicts)
In the batch step parser (`aggregates.py`, currently `args=list(s.get("args",
[]))`), accept positional args from `args` OR `arguments`. **Canonical key is
`arguments`** (execute_step parity). When both are present and equal, accept;
when both are present and DIFFERENT, return a validation error rather than
silently shadowing. Update the `execute_batch` docstring. Rationale: the bug is
a silent drop; arguments-first matches the rest of the API, and rejecting
conflicts avoids parity confusion (both reviewers).

### D6: Executable resolution hook against the server PATH (not shell)
Resolve a desktop launch/recovery executable to an absolute path via
`shutil.which` against the **server process** PATH (after the existing
`desktop_launch_env` sanitization) before dispatching `Process`/`Evaluate`,
and pass the absolute path. Do NOT inherit interactive shell startup state (a
security regression). Surface the effective PATH on resolution failure; allow
an optional config for extra desktop-tool paths. Document `Evaluate` as
expression-only and point to `Run Process` for statement-capable recovery.
Rationale: the report's `xdotool` mismatch is a resolution problem; a hook
fixes it without widening the server env (both reviewers; the `Evaluate`
subprocess inherits the server env, not a login shell).

### D7: Isolate pre-start steps via an explicit opt-in (default exclude)
Add `build_test_suite(include_pre_start: bool = False)`. Default **excludes**
pre-`start_test` steps from the generated test body and reports
`excluded_pre_start_count` + a summary in the response; `True` preserves the
prior adoption (`test_builder.py` adoption of `session.steps`) with an INFO
deprecation log. Update the `start_test` message to explain the handling.
Rationale: the report shows exploratory `Log` steps contaminating the suite;
the opt-in makes the previously-silent adoption explicit and reversible. (Open
question resolved: default is `False`.)

### D8: Live e2e reproduces the report flow end-to-end
Add a desktop e2e (under the isolation bootstrap) that runs the report's exact
flow — analyze(context=desktop) → init → discover PlatynUI keywords → inspect
state → stepwise real calculator interactions with per-entry + result
assertions → build_test_suite (real interactions) → run. Rationale: the user
asked to reproduce and fix; the e2e proves the cascade is broken.

## Risks / Trade-offs

- **Recommender reordering could affect non-desktop flows** → Gate the
  PlatynUI-first rule on desktop context / desktop signal only; leave web/
  mobile/api recommendations unchanged; unit-test both directions.
- **Batch `args`/`arguments` precedence ambiguity** → `arguments` is canonical;
  equal-both is accepted; conflicting-both is rejected as a validation error
  (no silent shadowing). Documented + tested both directions.
- **State-inspection routing change could regress web/mobile state** → Gate on
  `is_desktop_session()` strictly; web/mobile sessions keep their existing
  source path; unit-test the routing decision.
- **find_keywords fix scope** → Root cause is named (D4): library listing must
  work (D0) + `catalog` is a literal substring filter (server.py). Keep the fix
  scoped to PlatynUI discoverability + a documented fallback for NL queries.
- **Suite-isolation change could surprise callers who relied on adoption** →
  `include_pre_start=False` default excludes pre-start steps and reports the
  count; `True` preserves prior adoption with an INFO deprecation log. Explicit
  in response + docstring.
- **Resolution hook scope** → Resolve via `shutil.which` on the server PATH and
  pass the absolute path; do not import interactive shell startup state (a
  security regression). Optional extra-paths config is a follow-up.

## Migration Plan

1. Reproduction tests for the actionable findings (context param, batch args)
   as red unit tests.
2. D0 (register PlatynUI) — prerequisite for D2 + D4.
3. D1/D2 (context routing at all 3 sites + recommender) + tests, including the
   non-desktop regression guard.
4. D5 (batch args: arguments-first, reject conflicts) + test; D3 (state
   inspection routing) + test.
5. D4 (keyword discovery: library listing + NL-query fallback).
6. D6 (resolution hook + Evaluate docs); D7 (`include_pre_start=False`).
7. D8 live e2e of the full report flow under the isolation bootstrap.
8. ADR + acceptance finding-matrix (findings 1–10 → D0–D8 → status → test);
   report cross-reference; release notes.
9. Rollback: each fix is additive/gated (desktop context, desktop session,
   batch arg compat is backward-compatible; `include_pre_start=True` restores
   prior adoption); non-desktop and existing `args`-based callers are
   unaffected.

## Open Questions

- D7 default resolved: `include_pre_start=False`.
- D6 boundary resolved: server-process PATH + resolution hook (not shell
  inheritance); the optional extra-paths config is a follow-up if demand
  appears.
- D4: keep `catalog` literal + add a documented semantic fallback, or change
  catalog semantics? (Lean: keep literal, ensure library-listing works, and
  guide to semantic search — least surprising contract change.)
- Sequencing: this change must land AFTER `platynui-desktop-safety-isolation`
  (the Process allow-list + desktop classification) so the full
  analyze→init→Process flow is testable end-to-end.
