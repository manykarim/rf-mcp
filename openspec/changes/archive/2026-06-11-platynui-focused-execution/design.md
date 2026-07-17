## Context

rf-mcp's PlatynUI new-core integration (ADR-025) drives native desktop apps
through their accessibility tree. PlatynUI pointer/keyboard operations are
delivered at **screen coordinates** via the platform input backend (XTest on
X11/XWayland, EIS/portal on Wayland). The accessibility tree can be fully
addressable even when the AUT window is **not the focused/topmost window** —
and on overlapping windows the same screen coordinate belongs to whichever
window is stacked on top. The result, observed repeatedly during ADR-025
cross-agent validation:

- Operations "succeed" (the keyword returns OK) while input goes to a
  different active window, or to nothing visible.
- The "stacking trap": pre-launching two apps leaves the later one on top;
  the first scenario's clicks hit the wrong window.
- Agent-spawned MCP servers sometimes run with a stripped environment (no
  session bus / display), so launched apps don't even register on AT-SPI.

Current desktop sessions already skip web-style pre-validation and timeout
injection, and force `XDG_SESSION_TYPE=x11` to avoid the Wayland portal hang.
What is missing is a **focus-before-act** guarantee and a **visibility
precondition** so operations provably target the AUT, plus a reproducible
cross-agent harness to validate it.

Constraints: PlatynUI is an optional integration (Python ≥3.12); the
WindowSurface activation pattern requires a real WM/EWMH and is unavailable on
WM-less X servers (Xvfb) and on pure Wayland; the upstream `resolve_extents`
coordinate fix (ADR-025) is required for correct targeting on WM-less X.

## Goals / Non-Goals

**Goals:**
- Make focus-before-act the default for DESKTOP_TESTING sessions so pointer/
  keyboard input cannot leak to a non-AUT window.
- Detect and report AUT window visibility; warn (default) or fail-fast on
  operations against non-visible windows instead of silently passing.
- Verify a resolved interaction target belongs to the AUT window subtree
  (cross-window collision guard).
- Provide an explicit focus capability (intent + keyword) and guide agents to
  use it; keep all behavior overridable via an escape hatch.
- Deliver a reproducible multi-CLI (Codex/OpenCode/Kilo/Claude) experiment
  harness + a single findings report.

**Non-Goals:**
- Implementing a window manager or compositor. We raise/focus existing
  windows; we do not manage tiling/stacking policy.
- Changing web/mobile/API session behavior.
- Guaranteeing pixel-correct input on pure-Wayland-only environments without
  XWayland (documented as a limitation; X11/XWayland is the supported path).
- Adding a new Python runtime dependency.

## Decisions

### D1: Focus-before-act enforced at the executor desktop chokepoint
Insert the focus/visibility/scope checks in
`keyword_executor` at the same desktop chokepoint that already gates
pre-validation/timeout skips, immediately before dispatching a PlatynUI
pointer/keyboard keyword. Rationale: one chokepoint covers direct
`execute_step`, `intent_action`, and batch paths uniformly; mirrors the
existing ADR-025 desktop-skip design. Alternative considered: doing it inside
the plugin's `before_keyword_execution` hook — rejected because that hook
fires before argument/descriptor resolution and does not see the resolved
target element needed for the window-subtree check.

### D2: Window resolution via PlatynUI node ancestry, not WM queries
Resolve the AUT top-level window by walking the resolved node's ancestors to
the nearest WindowSurface-capable node (`top_level_or_self` / window-surface
pattern), using PlatynUI's own model. Rationale: works identically across
backends and avoids coupling to X11/EWMH. The WM is only consulted as a
*focus mechanism*, never as the source of truth for "which window".

### D3: Tiered focus strategy with portable fallback
Focus is attempted in order: (1) PlatynUI WindowSurface activate/raise when
available; (2) portable raise/focus fallback when WindowSurface is missing or
raises a pattern error (the WM-less case). The fallback is implemented behind
a small internal strategy interface so the X11 `XRaiseWindow`-class mechanism,
a Wayland path, and a no-op (already-focused) can be selected by environment.
Rationale: ADR-025 proved WindowSurface raises `PatternError` under Xvfb;
operations must still target the AUT. Alternative considered: always shelling
to `wmctrl`/`xdotool` — rejected as an external dependency and X11-only.

### D4: Visibility from accessibility attributes + desktop bounds
Compute visibility from `IsVisible`/`IsInView`/`Bounds` of the AUT window
intersected with `desktop_info` bounds, exposed in the `ui_tree` section.
Rationale: no new dependency, consistent with how ui_tree is already built.
Default policy is **warn**; **fail-fast** is opt-in per-call
(`fail_on_hidden=True`) and per-session. Rationale: warn-by-default avoids
breaking advanced/headless flows while still killing the silent false
positive; fail-fast is available where correctness must be enforced.

### D5: Window-subtree scope check is advisory-by-default, strict opt-in
After resolving the target element, verify its `top_level_or_self()` equals
the AUT window resolved from the descriptor's app scope. On mismatch: warn by
default, error when strict mode is on. Rationale: app-scoped descriptors
(`/app:*[@Name='X']//...`) already make true cross-app collisions rare; the
check is the safety net for unscoped descriptors and overlapping same-app
windows. Pair with guidance that discourages unscoped `//`.

### D6: Focus capability surfaced as intent verb + documented keyword
Add an "ensure focused / activate window" intent that maps to the PlatynUI
focus path for desktop sessions, and document the existing window keywords
(`Activate Window`, `Bring To Front`) plus the focus default in the
`platynui` locator-guidance topic and MCP instructions. Rationale: agents
need an explicit, discoverable way to focus at test start and on app switch;
the default covers the common case, the explicit verb covers deliberate
control.

### D7: Escape hatch via per-call arg + env var
`focus=False` (per-call) and `ROBOTMCP_PLATYNUI_NO_FOCUS=1` (process) disable
focus-before-act; the step result records `focus_bypassed=True`. Mirrors the
existing `ROBOTMCP_PLATYNUI_KEEP_WAYLAND` opt-out style. Rationale:
consistency and provenance.

### D8: Experiment harness as a documented, scripted matrix
The harness is a scripted runner (not a pytest gate — the agent CLIs are
external, rate-limited, and non-deterministic) that: stages an isolated
display, launches the AUT visibly, invokes each CLI with the canonical
scenario prompt + a normalized MCP-server env (display, session type, GDK
backend, `DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `HOME`), captures the
transcript + verdict, and aggregates a findings report. Per-CLI adapters
encode each tool's MCP-config mechanism and approval/sandbox flags (e.g.
Codex requires sandbox bypass for MCP calls and `< /dev/null`). Rationale:
keeps the live multi-agent runs reproducible and out of the deterministic
unit/CI path while still being repeatable on demand.

## Risks / Trade-offs

- **Focus thrash / latency** (focusing before every op adds raise+focus
  cost) → Cache the resolved AUT window per (session, app) and skip the raise
  when the AUT is already the active/topmost window; only re-assert on app
  switch or when the active-window check fails.
- **Raise/focus has side effects on the user's real desktop** (could steal
  focus) → Default experiment/e2e runs target an isolated Xvfb; document that
  focus-before-act on a shared desktop will raise the AUT. Escape hatch
  available for users who manage focus themselves.
- **Pure-Wayland focus is not guaranteed** (no portable raise without a
  compositor protocol) → Document X11/XWayland as the supported path; on
  Wayland, fall back to WindowSurface where the compositor supports it and
  warn otherwise.
- **WM-less environments cannot truly "focus"** (no input focus model) → On
  Xvfb the raise + PointerRoot semantics suffice for XTest delivery; the
  window-subtree scope check is the real guarantee there. Covered by D5.
- **False window-scope warnings** on legitimate multi-window apps (dialogs,
  popovers) → The check uses `top_level_or_self` and treats child dialogs of
  the same application as in-scope; strict mode is opt-in to avoid noise.
- **Agent CLIs change flags/behavior** (external tools) → Per-CLI adapters
  isolate the coupling; the harness records skips when a CLI can't be driven.
- **Visibility heuristic edge cases** (compositor reports stale geometry after
  a move) → Documented; recompute on demand and prefer the
  active-window check over cached geometry for the focus decision.

## Migration Plan

1. Land focus/visibility/scope logic behind the desktop chokepoint with the
   default = warn (non-breaking for passing-but-hidden runs except they now
   carry a warning). Ship the escape hatch in the same change.
2. Add the focus intent verb + guidance/instructions updates.
3. Add unit tests (focus resolution, fallback selection, visibility
   computation, scope check) and live e2e assertions that operations only
   affect the AUT window (extending the existing GNOME e2e suite).
4. Build and run the multi-CLI experiment harness; publish the findings
   report; feed any new issues back as follow-up tasks.
5. Rollback: the entire behavior is gated on DESKTOP_TESTING + a default that
   can be reverted to direct-dispatch via `ROBOTMCP_PLATYNUI_NO_FOCUS=1`;
   non-desktop sessions are unaffected, so rollback is low-risk.

## Open Questions

- Should fail-fast (not warn) become the default for visibility once the
  cross-agent harness confirms no legitimate flow depends on
  non-visible-but-addressable operation? (Lean: yes, after one validation
  cycle.)
- OpenCode CLI and Kilo CLI MCP-config + approval mechanics are not yet
  characterized in this repo (only Claude and Codex are) — the harness's
  first run will determine whether they need per-CLI env normalization
  beyond the documented set.
- For multi-window applications, do we need a user-supplied "primary window"
  hint, or is nearest-WindowSurface-ancestor sufficient in practice?
