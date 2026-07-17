## Context

An 8-run live investigation (Codex agent → robotmcp → GNOME apps) plus direct
probing established that the desktop read/locate/suite-build path works
end-to-end on an app that exposes AT-SPI (gnome-text-editor resolved app
root/Frame/Text; the agent built a fresh suite with real `Should Be*`), and
isolated three robotmcp gaps that silently hid hard failures:

- `Pointer Click`/`Keyboard Type` returned `OK` but text never entered
  (`CharacterCount` stayed `0`). The pure helper `input_effect_hint`
  (`desktop_execution_signals.py`, ADR-029 D2b) exists but is NOT wired.
- Forced X11 (`ensure_x11_session_env`) + XTest input is blocked by mutter on a
  real Wayland session; `ensure_x11_session_env` already detects the Wayland
  origin (it forces X11 when `XDG_SESSION_TYPE=wayland`/`WAYLAND_DISPLAY` set)
  but doesn't warn.
- A missing `XAUTHORITY` made the broker fail to connect to `:0`; `get_runtime`
  returned None and `get_ui_tree` reported "platynui-native not installed or
  runtime unavailable". The native module is ONE-SHOT (`platynui_plugin` broker
  states: new|open|shutting_down|disposed; disposed is terminal) so recovery
  needs a process restart — but nothing said so.

## Goals / Non-Goals

**Goals:**
- Turn each silent/misleading failure into a clear, actionable signal.
- Stay best-effort, non-reentrant, and off the happy path.

**Non-Goals:**
- Making Wayland synthetic input work (architectural — would need libei/eis).
- Re-initializing the one-shot native runtime (impossible in-process) — we
  diagnose and say "restart", not recover.
- Changing read/query behavior, the happy path, or non-desktop sessions.

## Decisions

### D1: Auto-wire input-effect detection (non-reentrant)
In the desktop branch of `_execute_keyword`, for a keyboard/typing interaction
whose target text node is resolvable, snapshot `native:Text.CharacterCount` via
the shared native `Runtime` DIRECTLY (`get_runtime().evaluate(...)`) before and
after the keyword — NOT by re-entering RF keyword execution (which holds the
`_execution_lock`; the reason ADR-029 left this unwired). Pass the snapshots to
the existing `input_effect_hint`; append a `desktop_input_no_effect` warning on
success+unchanged. Best-effort, guarded, skipped when no snapshot. Scope to
keyboard keywords first (CharacterCount is the reliable signal for typing);
pointer-only effects are app-wide and out of scope for the first cut.

### D2: Wayland-input warning
`ensure_x11_session_env` records the ORIGINAL session type (Wayland vs X11)
before forcing X11 — e.g. a module flag `was_wayland_session()` /
`session_origin()`. A small helper `wayland_input_warning(keyword, origin)`
(in `desktop_execution_signals`) returns the `wayland_x11_input_blocked_risk`
hint for an interaction keyword on a was-Wayland session. Wire it on the first
desktop interaction keyword per session (a session flag so it fires once, like
the tree-dirty flag). Read/query keywords do not warn.

### D3: Classify runtime failures + actionable diagnostic
The broker records the last bind exception and classifies it
(`runtime_unavailable_reason()` → `not_installed` | `display_connect_failed` |
`disposed` | None). `get_runtime`'s except-branch stores the reason; the
disposed branch maps to `disposed`. `ui_tree_service`'s `runtime is None` branch
(and any other runtime-None response) builds a structured diagnostic from the
reason: `display_connect_failed` → "failed to connect to the display — check
DISPLAY/XAUTHORITY/XDG_RUNTIME_DIR; the native runtime is one-shot, RESTART the
MCP server"; `not_installed` → the existing install hint; `disposed` → "restart
the MCP server". Never raises.

## Risks / Trade-offs

- **Input-effect snapshot adds per-keyboard-step native calls** → Only for
  keyboard keywords with a resolvable text node, two cheap `evaluate` reads,
  best-effort, guarded; skip on any error. Non-reentrant (native, not RF).
- **CharacterCount may be unreliable on some builds** (it was `0` on
  gnome-text-editor under XWayland) → That is itself the no-effect signal here;
  but to avoid false positives when CharacterCount is simply unavailable, only
  warn when a BEFORE snapshot was obtained and equals the AFTER snapshot (both
  readable, unchanged) — if the attribute is unreadable, skip (no warning).
- **Wayland warning could be noisy** → Fire ONCE per session (flag), only on
  input-injecting keywords; clearly advisory.
- **Runtime classification from error strings is heuristic** → Match
  conservatively (connection/display/shutdown → display_connect_failed; import
  → not_installed); default to a generic "unavailable" reason when unsure; never
  raise.
- **No live display/AT-SPI/Wayland in CI** → Unit-test with the native runtime,
  the signals, and `ensure_x11_session_env` origin mocked.

## Migration Plan

1. D3 broker reason classification + `runtime_unavailable_reason()` + ui_tree
   diagnostic; tests (mocked).
2. D2 Wayland-origin capture + warning helper + first-interaction wiring; tests.
3. D1 input-effect snapshot wiring (native, non-reentrant) + tests.
4. Full suite green; ADR + release note.
5. Rollback: each is additive/gated/best-effort; revert independently.

## Open Questions

- D1: extend the no-effect check to pointer clicks (app-wide state) later, or
  keep it keyboard-only? (Lean: keyboard-only first cut; pointer is ambiguous.)
- Land after ADR-032 (`desktop-native-platynui-alignment`).
