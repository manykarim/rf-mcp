## Context

Source of record: `docs/gnome-calculator-mcp-maintainer-report.md` (rewritten
after the `desktop-mcp-workflow-correctness` fixes) plus the captured trace
`tests/e2e/gnome_calculator_mcp_stepwise_trace.robot`. The trace is a real
agent session that classifies desktop correctly and launches the app, then
spends 50+ steps on raw `Evaluate` introspection of `UiNode` objects, overrides
the safety guard via `Evaluate ... os.environ[...] = '1'`, can't disambiguate
three duplicate `1` buttons, observes input keywords "succeed" with no display
change, and finally builds a debugging-trace suite.

Reproduced on the current branch:
- **Finding #9 (OBS-11 false substitution).**
  `test_builder._propagate_assigned_variables_to_literal_args` rewrites a
  later literal `"1"` into `${active_desktop_override}` because the override
  step captured the string `"1"`. Confirmed by a direct call:
  `Keyboard Type ${None} 1` → `Keyboard Type ${None} ${active_desktop_override}`.
- **Finding #2 (dead snap-sanitization).**
  `keyword_executor._maybe_sanitize_desktop_launch` builds the snap-decontaminated
  env via `build_desktop_launch_env`, but `grep` confirms it is never called on
  the execution path — a snap-confined `Start Process` therefore inherits the
  contaminated env and can exit `127`.
- **Finding #10.** `_INSPECTION_ONLY_KEYWORDS` does not include `Query`/`Evaluate`,
  and the `assign_to` carve-out preserves any step with an assigned variable —
  so the agent's `assign_to`'d exploratory probes are all serialized into the
  suite.
- **Finding #3.** The refuse message
  (`desktop_display_safety.py`) names only the
  `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP` bypass, with no isolation recipe.

## Goals / Non-Goals

**Goals:**
- A desktop GUI `Start Process` is snap-decontaminated (finding #2).
- "Input succeeded" is distinguishable from "the AUT reacted" (finding #4); a
  dead-process-but-live-nodes mismatch is surfaced (finding #2).
- The active-desktop refusal gives an actionable isolation path (finding #3).
- Agents read `UiNode` attributes, disambiguate duplicate roots/controls, learn
  symbol-vs-word naming, and read display state without `Evaluate` guessing
  (findings #5–#8).
- The generated suite reflects executed intent: recorded args are faithful
  (finding #9) and exploratory introspection is filtered (finding #10).
- Explicit `context="desktop"` is deterministic and documented (finding #1).

**Non-Goals:**
- Re-deriving desktop routing/recommendation (shipped by
  `desktop-mcp-workflow-correctness`) or the safety classifier itself (shipped
  by `platynui-desktop-safety-isolation`).
- Making GTK4 expose entry text content over AT-SPI (an upstream limitation);
  we document the `CharacterCount` length proxy, not change the toolkit.
- Hardening the guard against a determined `Evaluate`-based env bypass — finding
  #4 shows the bypass does not even make input work, so the value is a guided
  isolation path, not an un-bypassable gate.

## Decisions

### D1: Wire the snap-sanitized desktop launch (finding #2)
Call `_maybe_sanitize_desktop_launch` on the desktop branch of
`_execute_keyword` (next to the existing focus/resolution hooks), so a
desktop-session `Start Process`/`Run Process` of a known GUI binary gets the
snap-decontaminated `env:` overrides. Reuses the already-implemented
`build_desktop_launch_env`; the only change is activating it. Gate strictly on
`is_desktop_session()` + GUI-binary detection so non-desktop/non-GUI launches
are untouched.

### D2: Surface Process/discovery + input-effect disagreement (findings #2, #4)
- After a desktop `Start Process`, when a quick liveness check shows the process
  is not running, attach a warning (the visible PlatynUI nodes may be a
  different/stale instance) rather than implying a live AUT.
- For a desktop pointer/keyboard interaction, take a bounded before/after probe
  of the target's accessible display state (e.g. `native:Text.CharacterCount`)
  and, when the keyword returns success with no change, attach a soft warning.
  Best-effort and Browser-style (mirrors the existing PostActionVerifier
  pattern); never fails the step.

### D3: Actionable isolation recipe on refusal (finding #3)
Extend the refuse payload from `desktop_display_safety` with a concrete
isolation recipe (Xvfb/`systemd-run` bootstrap + the
`ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY` marker) and reframe the bypass env var as
an escape hatch that does not guarantee correct input on a shared desktop.

### D4: Driveability guidance (findings #5–#8)
Extend `get_platynui_locator_guidance` with: the `UiNode` attribute API (no
`get_attribute`; use `attribute()`/`attributes()`+`value()`, `name`, `role`,
or the `Get Attribute` keyword); duplicate-root/control disambiguation (multiple
live roots; scope with `Set Root`, prefer the interactable node); symbol-vs-word
control naming (operator `Button[@Name='+']` vs `Label[@Name='plus']`); and
display-state reading (`native:Text.CharacterCount` length proxy bound to the
entry `Text` node). The intent is to remove every reason the trace reached for
`Evaluate`.

### D5: OBS-11 propagation guard (finding #9)
In `_propagate_assigned_variables_to_literal_args`: (a) skip captures whose
value is shorter than a minimum length (single character/digit — coincidental
match risk dominates), and (b) skip captures sourced from an
introspection/`Evaluate` step. Keep the substitution for distinctive
multi-character data dependencies. This is the surgical fix for the
`Keyboard Type ${None} 1 → ${active_desktop_override}` corruption.

### D6: Stepwise-suite hygiene (finding #10)
Classify exploratory desktop introspection (`Query`, and `Evaluate` expressions
used only to inspect nodes) as non-recorded for suite generation by default,
EXCEPT when the step is load-bearing — its assigned variable is consumed by a
retained later step (assertion or interaction). Report the filtered count.
Compose with the existing `_INSPECTION_ONLY_KEYWORDS` + `record` gate; the new
piece is dependency-aware filtering so the suite still compiles.

### D7: Deterministic explicit desktop context (finding #1)
`desktop-mcp-workflow-correctness` already forces desktop on explicit
`context="desktop"` at every routing site; this change adds determinism tests
across phrasings and documents the guarantee in the `analyze_scenario`
guidance, closing the maintainer's "nondeterministic/prompt-sensitive" concern.

### D8: Validate end-to-end
An e2e assertion (under the Xvfb isolation bootstrap) that the suite generated
from the report flow is a clean validated calculator test — interactions +
assertions, no exploratory `Query`/`Evaluate` probes, faithful recorded args.

## Risks / Trade-offs

- **Activating the dead sanitizer could change desktop launch behavior** →
  Gate strictly on desktop + known GUI binary; honor the existing
  `platynui_no_sanitize` opt-out; unit-test the GUI/non-GUI split.
- **Input-effect verification adds per-step desktop work** → Bounded probe,
  best-effort, soft-warning only; never fails or blocks the step; desktop-only.
- **Suite-hygiene filtering could drop a needed step** → Dependency-aware:
  retain any introspection step whose assigned variable a kept step consumes;
  report the filtered count so nothing is silently lost; unit-test the
  load-bearing carve-out.
- **OBS-11 guard could miss a real single-char data dependency** → Single-char
  data dependencies are rare and ambiguous; the guard favors recording fidelity
  (keep the literal) over a speculative `${VAR}`; multi-char propagation is
  unchanged.

## Migration Plan

1. Reproduction tests: OBS-11 single-char false substitution (red), dead-sanitize
   wiring (assert the hook is called on a desktop GUI launch).
2. D5 (OBS-11 guard) + D1 (wire sanitizer) — smallest, highest-confidence.
3. D6 (suite hygiene, dependency-aware) + tests.
4. D2 (Process/input-effect signalling), D3 (isolation recipe), D4 (guidance).
5. D7 determinism tests + docs.
6. D8 e2e clean-suite assertion under the isolation bootstrap.
7. ADR mapping findings 1–10 → fixes; report + trace cross-reference; release notes.
8. Rollback: each fix is additive/gated (desktop-only sanitizer; soft warnings;
   OBS-11 guard only narrows substitution; hygiene filtering reports drops).

## Open Questions

- Input-effect verification (D2b) resolved during apply: the decision is a PURE,
  unit-tested helper (`desktop_execution_signals.input_effect_hint`). Automatic
  before/after probing is NOT wired inside `_execute_keyword` — a re-entrant RF
  query under the held execution lock risks deadlock and can't be validated in
  CI without a live desktop. D2a (launch liveness) IS wired non-reentrantly via
  the process handle's own `poll()`. A non-reentrant runtime-broker probe to
  auto-supply D2b snapshots is a follow-up.
- Suite hygiene: filter `Evaluate` entirely by default, or only `Evaluate`
  expressions with no consumed assignment? (Lean: filter unless load-bearing.)
- This change must land AFTER `desktop-mcp-workflow-correctness` and
  `platynui-desktop-safety-isolation`.
