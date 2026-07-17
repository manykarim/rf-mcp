# Proposal: desktop-isolation-marker-hardening

## Why

The desktop safety guard's job is to refuse synthetic input on a display it
cannot prove is isolated, so an agent never sprays clicks/keystrokes onto the
user's real session. The guard grants ISOLATED purely from an **environment
marker**, short-circuiting before the live probe:

- `classify_bound_display_detailed` (`desktop_display_safety.py:136-165`)
  returns `ISOLATED` at line 149 when `_has_isolation_marker(env, display)` is
  true — **before** `_ewmh_wm_present` runs. The docstring is explicit:
  "Positive isolation proof (the marker) takes precedence over the EWMH probe."
- `_has_isolation_marker` (`:48-60`) does verify the bound `DISPLAY` is listed
  in `ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY`, so the risk is narrower than an
  unconditional bypass — **but** it is still trust-on-assertion: if the marker
  names the currently-bound display value (e.g. `:0`) and that display is in
  fact the user's active desktop, the guard allows input onto it. A stale
  marker (a recycled `:99` that is now something else), a misconfigured export
  (`ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=:0`), or a marker inherited into the
  wrong process all defeat the guard silently.

Why a naive fix is wrong: you cannot simply require "no EWMH window manager" as
the consensus, because a **legitimately isolated** display routinely runs a WM
— the reference docker harness is Xvfb `:99` + **fluxbox** (an EWMH WM). The
EWMH probe returns "WM present" for the good isolated case too. So the marker
exists precisely because "has a WM" is not evidence of "is the user's desktop."
The correct hardening is to make the marker *harder to assert by accident* and
to *surface the conflict* when marker and probe disagree — not to override one
with the other. This is a fail-closed correctness/safety issue for
running against several apps on a developer's real machine (eval synthesis
2026-07-17; risk R4, re-scoped from "consensus" to "ownership + observability"
after reading the code).

## What Changes

- **Verify marker ownership, not just marker presence.** Grant ISOLATED from
  the marker only when the marker is corroborated by an ownership proof that a
  stray inherited/exported value cannot fake. Candidate proofs (pick per
  `design`): the marker value carries a token that matches a value rf-mcp itself
  minted for the display it launched, and/or the bound display's Xvfb/nested-X
  was started by this process tree. Absent corroboration, the marker alone
  yields `unknown` (fail-closed), not `isolated`.
- **Surface marker-vs-probe conflict in session state.** When a marker claims
  the bound display AND the EWMH probe reports an active-desktop-shaped WM that
  is not the expected isolated WM, record `isolation_source =
  marker_over_active_wm` (distinct from the clean `marker` case) so an operator
  and the agent can see the guard is trusting an assertion that conflicts with
  the live probe — the diagnostic that would have caught a misconfigured
  `:0` marker.
- **Keep the legitimate isolated-WM case working.** A marker corroborated by
  ownership continues to classify ISOLATED even though fluxbox (or any WM) owns
  the display — the docker harness and any user-run nested-X isolated display
  keep passing. No behavior change for the correct configuration.

Out of scope: replacing the ctypes EWMH probe (a documented native gap,
`_ewmh_wm_present:66-84`, retained deliberately); the active-desktop opt-in
(`ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP`) which remains the explicit hatch;
the input-landing verdict (owned by `desktop-steering-confidence-gate`).

## Capabilities

### New Capabilities

- `desktop-isolation-marker-hardening`: the safety guard grants ISOLATED from
  the environment marker only when corroborated by an ownership proof a stray
  value cannot fake, fails closed to `unknown` otherwise, and records a distinct
  `marker_over_active_wm` provenance when the marker conflicts with a live
  active-desktop probe — while the legitimately isolated (WM-owning) display
  continues to pass.

### Modified Capabilities

- None (the existing safety-guard behavior lives in code, not an
  `openspec/specs/` capability; requirements are additive/hardening).

## Impact

- `src/robotmcp/components/execution/desktop_display_safety.py:48-60` —
  `_has_isolation_marker` gains ownership corroboration (or a sibling
  `_marker_ownership_verified`).
- `src/robotmcp/components/execution/desktop_display_safety.py:136-165` —
  `classify_bound_display_detailed` no longer returns ISOLATED on marker alone;
  emits `marker_over_active_wm` when marker + active-WM probe conflict; falls to
  `unknown` when the marker is uncorroborated.
- Marker minting: wherever rf-mcp launches/adopts an isolated display (docker
  entrypoint / nested-X helper) sets the corroboratable marker token — the
  harness `entrypoint.sh` / `claude_mcp.json` env is updated in lockstep so the
  reference isolated display still classifies ISOLATED.
- Tests: `tests/unit/` — marker naming the bound display WITHOUT corroboration
  → `unknown` (fail-closed); corroborated marker on a WM-owning display →
  `isolated` (fluxbox case preserved); marker + active-desktop-WM conflict →
  `marker_over_active_wm` recorded; no marker → existing EWMH behavior unchanged.
- Docs: `docs/desktop_docker_harness.md` AT-SPI/isolation checklist gains the
  corroboration note.
