## Why

A maintainer report (`docs/gnome-calculator-mcp-maintainer-report.md`) walked
the full MCP flow for "open GNOME Calculator, do calculations, assert each
value, build a suite from executed steps" and never reached a single real
calculator interaction. The root cause was that a Linux desktop scenario was
routed through the **mobile/Appium** path, and that misrouting cascaded into a
chain of workflow failures. The prior change `platynui-desktop-safety-isolation`
fixed text-based classification and the desktop library allow-list; this change
closes the remaining, independently-reproduced MCP-workflow gaps the report
exposed so an agent can actually drive a desktop app end-to-end.

Reproduced on the current branch: `context="desktop"` is ignored by
`analyze_scenario` (the NLP processor has web/api/mobile/database branches but
no `desktop`), and `execute_batch` silently drops step arguments passed as
`arguments` (the batch schema only reads `args`).

## What Changes

- **`context="desktop"` forces a desktop session across ALL routing sites.**
  `analyze_scenario` honors an explicit desktop context at every site that
  drives session creation — the NLP analysis (`nlp_processor.analyze_scenario`,
  which has no `desktop` branch today), the platform detector
  (`session_manager.detect_platform_from_scenario`, whose mobile keyword list
  includes the generic token "app"), and session auto-configuration
  (`configure_from_scenario`, which takes no context today) — so a Linux
  desktop scenario is never routed to mobile/Appium when the caller said
  desktop. Acceptance is measured on the **full `analyze_scenario` tool
  result** (session_type, imported libraries, search order), not just NLP
  output.
- **PlatynUI is a first-class registry library; recommendation prefers it for
  desktop.** Add a `PlatynUI.BareMetal` entry to the library registry/
  recommender (it is currently only a runtime plugin, absent from
  `recommend_libraries`), and rank it ahead of `AppiumLibrary` for desktop/
  GNOME scenarios with the session search order reflecting that. Library-name
  aliasing (`PlatynUI` → `PlatynUI.BareMetal`) works in discovery/recommend.
- **`get_session_state` inspects the desktop accessibility tree for desktop
  sessions.** A desktop session's state/`page_source` request uses the
  PlatynUI `ui_tree` path instead of the mobile-source lookup that reported
  "Failed to get mobile source: No application is open".
- **`find_keywords` surfaces PlatynUI desktop keywords.** The reported zero
  matches have two named root causes: `find_keywords(library_name=
  "PlatynUI.BareMetal")` should list the library's keywords (catalog mode), and
  the `catalog` strategy is a literal substring filter so a natural-language
  query like "get window find element ui tree" returns 0 even when PlatynUI is
  indexed. Fix: ensure PlatynUI keywords are listable by library (incl. the
  `PlatynUI` alias), and either keep `catalog` literal with a documented
  fallback to semantic search or specify the catalog query contract — agents
  must reach the desktop interaction keywords (Pointer Click, Keyboard Type,
  Query, Get Attribute, window keywords).
- **`execute_batch` accepts `arguments` as well as `args`, arguments-first.**
  Batch steps no longer silently lose their arguments when the caller uses the
  same `arguments` field that `execute_step` uses. When only one key is
  present it is used; when both are present and equal it is accepted; **a
  conflicting dual-specification is a validation error** (not silent
  shadowing). For `execute_step` parity, `arguments` is the canonical key.
  **(bug fix)**
- **Execution-environment consistency via a resolution hook (not shell
  inheritance).** Desktop `Process`/`Evaluate` launches resolve an executable
  to an absolute path via `shutil.which` against the **server process** PATH
  (after the existing sanitization) before dispatch, so a tool present for the
  server (e.g. `xdotool`) is found — without inheriting arbitrary interactive
  shell startup state. An explicit opt-in config can add desktop-tool paths.
  `Evaluate`'s expression-only limitation is documented with a
  statement-capable alternative (`Run Process`).
- **Stepwise-suite isolation (explicit opt-in).** `build_test_suite` gains an
  `include_pre_start: bool = False` parameter: by default, exploratory steps
  executed before `start_test` are **excluded** from the generated test body
  (the response reports the excluded count + a summary); `include_pre_start=
  True` preserves the prior adoption behavior. The `start_test` message
  explains the handling rather than only warning about adoption, so
  `build_test_suite` produces a real suite, not a `Log`-only placeholder.

## Capabilities

### New Capabilities
- `desktop-context-routing`: explicit `context="desktop"` forces a desktop/
  PlatynUI session; `recommend_libraries` prefers PlatynUI for Linux desktop.
- `desktop-state-inspection`: `get_session_state` uses desktop accessibility
  (ui_tree) for desktop sessions instead of the mobile-source lookup.
- `desktop-keyword-discovery`: `find_keywords` surfaces PlatynUI desktop
  interaction keywords for desktop sessions/queries.
- `batch-step-argument-compat`: `execute_batch` accepts both `args` and
  `arguments` for step positional arguments.
- `desktop-exec-environment`: consistent executable/PATH resolution for
  desktop launches and recovery; documented `Evaluate` expression limitation.
- `stepwise-suite-isolation`: pre-`start_test` exploratory steps are isolated
  from `build_test_suite` so the generated suite is not a placeholder.

### Modified Capabilities
<!-- The desktop classification + Process allow-list (change
     platynui-desktop-safety-isolation) and PlatynUI keyword surface
     (ADR-025/026) are not yet archived openspec capabilities, so these are
     new specs; this change composes with — and depends on — that work. -->

## Impact

- **Code**: `components/nlp_processor.py` (desktop context branch);
  library recommender (PlatynUI-first for desktop); `server.py`
  `get_session_state` / page-source routing for desktop sessions;
  keyword discovery/catalog for PlatynUI; `domains/batch_execution/`
  (`args`/`arguments` compat); `Process`/`Evaluate` launch env;
  `build_test_suite`/`start_test` step gating.
- **Behavior**: desktop scenarios route to PlatynUI; desktop state inspection
  and keyword discovery work; batch preserves arguments. No change to web/
  mobile/API routing when the caller does not request desktop.
- **Tests**: unit tests (context routing, recommender order, batch arg compat,
  page-source routing decision, discovery results) + a live desktop e2e that
  drives the report's scenario end-to-end (analyze → init → discover →
  inspect → stepwise interactions → build → run) under the isolation bootstrap.
- **Dependencies/env**: builds on `platynui-desktop-safety-isolation`
  (safety guard, isolation bootstrap, classification, Process allow-list) and
  ADR-025/026. No new Python dependency.
- **Docs**: a new ADR mapping each report finding to its fix; the maintainer
  report referenced as the source of record.
