## Context

Source: a live end-to-end run where an external Codex agent drove the robotmcp
MCP server against GNOME Calculator on an isolated Xvfb display. The scenario
completed (three calculations, per-entry + result assertions, a clean generated
suite), validating the recent `desktop-stepwise-execution-fidelity` and
`desktop-mcp-workflow-correctness` work, but exposed two reproduced issues.

Reproduced in code:
- **Finding 1.** `get_platynui_locator_guidance()["display_state_reading"]`
  leads with `native:Text.CharacterCount`. In the live run that attribute
  returned `0` on all six `Text` nodes while the display visibly changed; the
  agent succeeded only by reading the history `Label` nodes (and OCR).
- **Finding 2.** `plugin_manager.get_library_for_keyword("Start Process")`
  returns `None`. The `Process` library is a static `definitions.py` entry whose
  `StaticLibraryPlugin.get_keyword_library_map()` defaults to `None` (unlike the
  `RequestsLibrary`/`PlatynUI` plugins, which declare their keyword maps). With
  no mapping, `keyword_executor._ensure_library_registration` →
  `_get_library_for_keyword` returns `None` and never registers `Process`, so an
  unqualified `Start Process` is unresolved. `Process.Start Process` works via
  the dotted-name short-circuit.

## Goals / Non-Goals

**Goals:**
- The PlatynUI display-state guidance leads with the reliable history-Label
  path; CharacterCount is a flagged secondary proxy; OCR is the last resort.
- Unqualified `Process` keywords resolve and register so a desktop session can
  launch its AUT without dotted names.

**Non-Goals:**
- Fixing the upstream GTK4/AT-SPI `CharacterCount` behavior (we document and
  route around it, not change the toolkit).
- Adding keyword maps for every static RF library — scope to `Process` (the one
  the live run needed); the mechanism is reusable for others later.
- Changing web/api/mobile session library sets or resolution.

## Decisions

### D1: Lead display-state guidance with history Labels
Rewrite the `display_state_reading` section of `get_platynui_locator_guidance`:
1. PRIMARY — read the history/result `Label` nodes (named with the equation and
   the result value); this is what works on GNOME Calculator.
2. SECONDARY — `native:Text.CharacterCount` as a length proxy, explicitly
   flagged "may report 0 on some GTK builds even when the display changed; do
   not rely on it alone".
3. LAST RESORT — screenshot + OCR.
Also soften the cross-reference in the `node_attribute_api`/`tips` sections that
currently steer to CharacterCount as the display hook. Pure guidance/string
change; no behavioral risk.

### D2: Give the Process definition a keyword→library map
Extend the static-definition mechanism so the `Process` entry in
`plugins/builtin/definitions.py` declares its keyword set, and
`StaticLibraryPlugin.get_keyword_library_map()` returns
`{kw: <library-name>}` when the definition provides one. Populate the Process
keyword set (Run Process, Start Process, Get Process Id/Object/Result, Is
Process Running, Process Should Be Running/Stopped, Send Signal To Process,
Switch Process, Terminate Process, Terminate All Processes, Wait For Process,
Split/Join Command Line). Then `get_library_for_keyword("Start Process")` →
`Process`, and `_ensure_library_registration` loads + registers Process into the
RF context on first unqualified use. This is the minimal, general root-cause fix
— it works for any session, and is exactly what a desktop launch needs.
Alternative considered: eager-import Process only for desktop sessions at
session setup. Rejected as the primary fix because the resolver gap is the true
root cause and the map fixes it everywhere; an eager desktop import can be added
later if first-use registration latency is a concern.

### D3: Keep qualified names and non-desktop behavior intact
The dotted-name branch in `_get_library_for_keyword` is untouched, so
`Process.Start Process` still resolves directly. The keyword map only ADDS
resolutions; no existing mapping changes, so web/api/mobile sessions are
unaffected (they simply gain the ability to resolve unqualified Process
keywords too, which is harmless and consistent).

## Risks / Trade-offs

- **Process keyword names could collide with another library's keywords** →
  Process keyword names (Start Process, Run Process, …) are distinctive and not
  shared by Browser/Selenium/PlatynUI/BuiltIn; the map is additive and the
  dotted-name + explicit-preference paths still win. Unit-test the resolutions.
- **Guidance-only change for D1 has no test of real behavior** → That is
  inherent (it is guidance); we test that the structured guidance now leads with
  Labels and flags CharacterCount, which is the observable contract.
- **First-use registration latency for Process** → One-time load on the first
  unqualified Process keyword; negligible and identical to how other on-demand
  libraries load.

## Migration Plan

1. D2 keyword-map mechanism + Process keyword set; unit test
   `get_library_for_keyword("Start Process") == "Process"` and the desktop
   registration path.
2. D1 guidance rewrite; unit tests on the structured guidance ordering + flags.
3. Regression: non-desktop sessions unchanged; qualified names unchanged.
4. ADR + reference to the live-run evidence; release note.
5. Rollback: both changes are additive (a guidance string + an additive keyword
   map); revert either independently.

## Open Questions

- Should the keyword-map mechanism be extended to OperatingSystem/Collections/
  String now, or only Process? (Lean: Process now; generalize on demand.)
- This change composes with `desktop-stepwise-execution-fidelity` (ADR-029);
  land after it.
