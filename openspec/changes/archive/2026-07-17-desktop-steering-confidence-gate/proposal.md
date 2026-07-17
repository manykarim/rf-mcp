# Proposal: desktop-steering-confidence-gate

## Why

The most dangerous instability in agentic desktop steering is not slowness — it
is **false success**: an interaction keyword returns `PASS` while the keystroke
or click never reached the application (lost to an unfocused, hidden, or foreign
window, or dropped by a Wayland compositor). The agent reads "success," moves
on, and validates nothing. rf-mcp already *computes* the signals that would
catch this, but emits them as soft hints the model can (and does) ignore:

- **`input_effect_hint`** (`desktop_execution_signals.py:172-205`) already
  compares an accessible-state snapshot (e.g. `native:Text.CharacterCount`)
  before/after a successful interaction and returns a
  `desktop_input_no_effect` hint when nothing changed — but it is advisory only.
- **Focus is tracked but not enforced.** `FocusOutcome`
  (`platynui_focus.py:184`) carries `warnings`; `has_verified_focus()`
  (`:472`) reports whether activation was actually verified; the visibility
  precondition (`:323-348`) yields warnings like `window IsVisible=false` /
  `off-screen`. Yet `ensure_focused(..., strict_scope=False, fail_on_hidden=…)`
  (`:636-643`) defaults to **not** failing — cross-window collisions and
  invisible-window keystrokes warn and proceed.
- **Wayland input-drop is a warning too.** `wayland_input_warning`
  (`:138-160`) returns a `wayland_x11_input_blocked_risk` hint; nothing
  enforces it. The docker harness only dodges this by forcing X11.

Evidence this is a live failure class, not a hypothetical: the documented
2026-06-11 LibreOffice silent-keystroke-loss (reader 1), the
`type-at-focus with no previously verified AUT window focus` warning string
(`:61-63`) that exists precisely because keystrokes vanished before, and the
`desktop_input_no_effect` message itself ("Success is not evidence the
application reacted"). Across multiple apps and toolkits — exactly the target
of this work — the probability of a hidden/foreign/unfocused target rises
(modals, multi-window, slow LibreOffice bring-up), so an enforced landing check
is what makes steering *trustworthy*, not just *attempted* (eval synthesis
2026-07-17, top-3 #2; risks R3+R7).

## What Changes

- **Compose a `steering_confidence` verdict for interaction keywords.** After an
  interaction step, combine the already-computed signals — verified-focus state
  (`has_verified_focus` / `FocusOutcome.warnings`), the visibility precondition
  warnings, `input_effect_hint`, and `wayland_input_warning` — into a single
  structured verdict attached to the step result: `confirmed` (focus verified
  and/or input effect observed), `unconfirmed` (no positive evidence either
  way), or `contradicted` (success reported but focus unverified AND input
  effect absent, or a Wayland drop-risk on an unverified target).
- **Fail-fast on `contradicted` by default.** A `contradicted` interaction
  SHALL be reported as a failure (not success) by default, because it is the
  provable "passed but did not touch the app" case. `unconfirmed` remains a
  success carrying the verdict (no positive proof of effect is available for
  every keyword/app — e.g. a button with no readable state). An opt-out
  (`ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE=warn`, mirroring the existing
  `ROBOTMCP_PLATYNUI_SAFETY_GUARD=warn`) downgrades enforcement to advisory.
- **Make the verdict machine-parseable and consistent.** The verdict replaces
  the scatter of ad-hoc `warnings` strings for the landing question with one
  field the agent must consult, so a weak model gets an unambiguous "the input
  did not land — refocus and retry" signal instead of a success it will trust.

Out of scope: the batch retry-safety gate (owned by
`desktop-aware-batch-execution` §5 — this change supplies the per-step verdict
that gate consumes); adding new read-back proxies for widgets that expose no
state (documented AT-SPI limitation); the `strict_scope` wrong-window *scope*
policy remains available and unchanged (this change targets the landing/effect
question, and reuses its warnings as inputs).

## Capabilities

### New Capabilities

- `desktop-steering-confidence-gate`: every desktop interaction keyword carries
  a structured `steering_confidence` verdict derived from verified-focus,
  visibility, input-effect, and Wayland-drop signals; a `contradicted` verdict
  fails the step by default (opt-out to warn), so "success" means the input
  demonstrably reached the application or its effect could not be disproven.

### Modified Capabilities

- None (no existing steering/focus capability spec under `openspec/specs/`;
  requirements are additive and reuse existing signal producers).

## Impact

- `src/robotmcp/components/execution/desktop_execution_signals.py` — a
  `steering_confidence(...)` composer over the existing `input_effect_hint`,
  `wayland_input_warning`, and focus/visibility warning inputs; verdict enum +
  `warn` opt-out reader (mirror `desktop_display_safety.warn_mode`).
- `src/robotmcp/components/execution/platynui_focus.py:184-473` — expose the
  verified-focus / visibility signals to the composer (already computed on
  `FocusOutcome`); no change to the raise tiers.
- `src/robotmcp/components/execution/keyword_executor.py` — at the interaction
  keyword result boundary, capture before/after state snapshot (reuse the
  existing `input_effect_hint` snapshot path), call the composer, attach the
  verdict, and raise on `contradicted` unless opted to `warn`.
- Tests: `tests/unit/` — `contradicted` (unverified focus + no CharacterCount
  change) fails by default and passes under `=warn`; `confirmed` (verified
  focus or observed effect) passes with the verdict; `unconfirmed` (no readable
  state) passes carrying the verdict; non-interaction keywords are unaffected.
- Deterministic validation (docker, no-LLM) — closes eval gaps G3/G4: minimize
  or lower the calculator, dispatch a keystroke, and assert the step is reported
  `contradicted`/failed (CharacterCount unchanged) rather than success.
