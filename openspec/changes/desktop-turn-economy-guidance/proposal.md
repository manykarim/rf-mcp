# Proposal: desktop-turn-economy-guidance

## Why

The desktop turn-economy spike (`experiments/SPIKE_2_desktop_efficiency.md`,
2026-07-10, Claude Code + MiniMax-M3 in the `robotmcp-desktop-lab` Docker)
measured that desktop runs burn **~60% of tool calls on discovery, not
actions** — stable across three independent runs (cc-desk-base 59%, desk-calc
61%, desk-edit 61%). The sinks this change targets:

- **Keyword discovery is the #1 sink** (spike §2.1): `find_keywords` semantic
  strategy returned nothing PlatynUI-specific in 4/4 desk-edit queries; the
  pattern strategy invited a one-guess-per-call loop — cc-desk-base made **13
  consecutive `find_keywords` calls** (calls 5-17), then 4 `get_keyword_info`
  calls, then **read the PlatynUI library source with wc/head/sed via Bash**
  (5 calls). PlatynUI keywords are not in any LLM's pretraining, so every gap
  in upfront guidance is paid for in round-trips. The whole surface is **24
  keywords** (confirmed via `LibraryDocumentation("PlatynUI.BareMetal")`) and
  fits in one ~2.2 KB (~550-token) payload — measured, not estimated.
- **No desktop upfront guidance exists** (spike §2.2):
  `src/robotmcp/domains/instruction/templates/` has `browser_focused.txt` and
  `api_focused.txt` but no desktop template, and `InstructionTemplateType`
  (`fastmcp_adapter.py:26-36`) has no desktop member. `manage_session(init)`
  returns no keyword surface for desktop sessions (`server.py:3178-3188`). The
  opt-in `get_locator_guidance` PlatynUI guidance
  (`rf_native_type_converter.py:1682` ff.) is genuinely good — it documents
  verbatim the app-scoping rule, Frame-vs-Window, and the result-Label recipe
  that cc-desk-base spent 8 calls rediscovering — but **the cc-desk-base agent
  never called it**. Optional guidance is not delivered guidance.
- **A bare desktop init cannot launch an app** (spike §2.5): `Process` is only
  in `optional_libraries` for `DESKTOP_TESTING`
  (`session_models.py:585-586`), so cc-desk-base's `Start Process` failed with
  "No keyword found" and the agent detoured through `OperatingSystem.Run` +
  raw Bash (3-5 wasted calls). The `analyze_scenario` path already adds
  Process (`nlp_processor.py:549-553`) — evidence the intended desktop
  baseline includes it — but nothing guarantees agents go through
  `analyze_scenario`. Every desktop scenario starts with a launch.

The bound on the upside is already measured: a run whose prompt pre-stated the
keywords/locators (`experiments/cc-desk-base` vs `experiments/cc-desk-batch`,
spike §3) went from **70 tool calls / 410 s / $4.98 to 11 calls / 160 s /
$0.85 (6.4× fewer calls)** with identical task quality. A real agent cannot be
prompt-fed locators for arbitrary apps, so rf-mcp must supply the equivalent
knowledge at init time. This change covers spike recommendations **#1**
(desktop upfront-guidance bundle) and **#5** (Process → core).

## What Changes

- **Desktop init bundle** — `manage_session(action="init")` responses for
  desktop sessions (session type `DESKTOP_TESTING`, or `PlatynUI.BareMetal` /
  `PlatynUI` among the requested/loaded libraries) gain a
  `desktop_guidance` field containing:
  - a **keyword cheat-sheet**: all 24 `PlatynUI.BareMetal` keyword names with
    one-line signatures preserving argument order (critically
    `Take Screenshot(descriptor, filename=…, rect=…)` — the arg-order trap
    that cost a 30 s hang in desk-calc, spike §2.3), derived from the
    library's libdoc at first use and cached process-wide (single source of
    truth; no drift when the keyword surface evolves);
  - a **locator crib**: app-scoping (`/app:*[@Name='X']//…`, never bare `//`),
    `Set Root` once then relative queries, **Linux windows are
    `control:Frame` not `control:Window`**, launch-before-query ordering,
    the result-read-back recipe (`Get Attribute … control:Label … Name`),
    the `Take Screenshot` arg-order warning, and a pointer to
    `get_locator_guidance` for the full guidance. Content is distilled from
    the existing `get_platynui_locator_guidance`
    (`rf_native_type_converter.py:1682` ff.) so there is one authoritative
    source for the rules.
  - The bundle is bounded (≤ ~3 KB; measured 2.2 KB) and absent from
    non-desktop init responses.
- **Desktop instruction template** — a new `desktop-focused` template joins
  `browser-focused`/`api-focused`: a factory classmethod in
  `instruction/value_objects.py` (workflow: init with PlatynUI + Process →
  `Start Process` → `Query` the `control:Frame` → act with pointer/keyboard
  keywords → read back via `Get Attribute`; plus the locator rules above),
  registered in `InstructionTemplate.get_by_name` and
  `InstructionTemplateType` so `ROBOTMCP_INSTRUCTIONS_TEMPLATE=desktop-focused`
  selects it, with a mirror `templates/desktop_focused.txt`.
- **Process becomes core for desktop** — `DESKTOP_TESTING`
  `core_libraries` gains `Process` (moved out of `optional_libraries`,
  `session_models.py:585-586`), so the profile-driven loading path
  (`get_libraries_to_load`, `session_models.py:1280-1307`) loads it on a bare
  init. `PlatynUI.BareMetal` stays first in `core_libraries` — the
  intelligent search-order builder derives from that order
  (comment at `session_models.py:581-584`) and the desktop library must keep
  leading the search order.

Out of scope (sibling changes from the same spike, per §5): `Take Screenshot`
fail-fast and the Linux `control:Window` pre-dispatch guard (levers #2/#4 →
`desktop-screenshot-failfast`), desktop-aware `execute_batch` + argument
validation parity (levers #3/#8 → `desktop-aware-batch-execution`), the
desktop actionable-controls view and launch intent (levers #6/#7 →
`desktop-actionable-controls`).

## Capabilities

### New Capabilities

- `desktop-turn-economy-guidance`: desktop sessions receive the PlatynUI
  keyword surface and locator rules upfront — a bounded cheat-sheet + crib in
  the `manage_session(init)` response, a selectable desktop-focused
  instruction template, and `Process` loaded as a core desktop library — so
  agents act instead of rediscovering the library one round-trip at a time.

### Modified Capabilities

None — the change is additive: existing template selection, init responses for
non-desktop sessions, and non-desktop session profiles keep their behavior.

## Impact

- `src/robotmcp/server.py:3178-3188` — attach `desktop_guidance` to the init
  result for desktop sessions.
- `src/robotmcp/models/session_models.py:585-586` — `Process` into
  `DESKTOP_TESTING.core_libraries` (out of `optional_libraries`);
  `PlatynUI.BareMetal` remains first. Loading path already honors core libs
  (`get_libraries_to_load`, `session_models.py:1280-1307`).
- `src/robotmcp/domains/instruction/value_objects.py:648-742` — new
  `desktop_focused` factory + `get_by_name` registration.
- `src/robotmcp/domains/instruction/adapters/fastmcp_adapter.py:26-36` —
  `DESKTOP_FOCUSED = "desktop-focused"` enum member.
- `src/robotmcp/domains/instruction/templates/desktop_focused.txt` — mirror of
  the new template (the `.txt` files are reference mirrors; production content
  lives in `value_objects.py`).
- New module for the cheat-sheet/crib builder (e.g.
  `src/robotmcp/components/execution/desktop_guidance.py`) so `server.py`
  stays thin; crib text sourced from/aligned with
  `rf_native_type_converter.py:1682` ff.
- Tests: `tests/unit/test_desktop_turn_economy_guidance.py` — bundle presence,
  content (24 keywords, arg order, Frame-not-Window, `Take Screenshot`
  signature), size bound, absence on web/API init; template selection via
  name and env; desktop profile loads Process while PlatynUI leads the search
  order. Existing suites to keep green: `test_platynui_newcore_plugin.py`,
  `test_multi_test_session.py`, instruction-domain tests.
