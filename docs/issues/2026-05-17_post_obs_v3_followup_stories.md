# Follow-up user stories: post-OBS-10/11/12 (v3) benchmark findings

**Source benchmark**: `docs/benchmarks/2026-05-17_tricentis_v3_post_obs_10_11_12.md`
**Triggered by**: validation re-run of the Tricentis Obstacle Course after
OBS-10/11/12 shipped on `feature/obstacle-course-followup`.
**Date**: 2026-05-17 (third validation run of the day)
**Status**: ready for implementation
**Predecessors**:
- `docs/issues/2026-05-17_obstacle_course_stories.md` (OBS-01..09)
- `docs/issues/2026-05-17_post_obs_followup_stories.md` (OBS-10/11/12, OBS-13/14 proposed)

The v3 validation run confirmed that OBS-10 (Drag And Drop pre-scroll) and
OBS-11 (variable propagation) delivered on their primary acceptance metrics:
Sonnet's Obstacle 10 dropped from 27 → 13 calls, Haiku from 9 → 7. Variable
propagation rewrote `${ENTRY_COUNT}` / `${ORDER_ID}` correctly in Sonnet's
generated suite.

During that same run, three new findings surfaced — one a real defect
(OBS-12 hint exists but is unreachable from the path where the bug
actually fires), two a quality / UX papercut (false-positive warning,
ARIA text disambiguation).

## Story index

| ID | Title | Type | Priority | Effort | Source obstacle |
|---|---|---|---|---|---|
| OBS-15 | OBS-12 hint must also reach the pre-validation failure path | Bug | Med | S | 8 (Wait a moment) — Sonnet v3 |
| OBS-16 | `untracked_variables` warning false-positive after OBS-11 substitution | Bug (low-impact) | Low | S | 7 (And counting) + 3 (Not a table) — Sonnet v3 |
| OBS-17 | ARIA snapshot conflates visible page text with element metadata | UX | Low | S | 3 (Not a table) — Sonnet v3 |

Plus one open investigation (not yet a formal story — see "Open
investigations" at end of file): Haiku v3 propagated 1 of 3 captured
variables; the other two used literals despite OBS-11 being in effect.

---

## OBS-15 — OBS-12 hint must also reach the pre-validation failure path

**Type**: Bug · **Priority**: Med · **Effort**: S

### Background

OBS-12 added a focused hint that fires when an agent uses a
SeleniumLibrary-style role-prefix locator (`button=X`, `link=X`,
`input=X`, `select=X`, `textarea=X`) against the Browser library. The
hint points at the working alternatives (`text=X` and
`css=<tag>:text-is('X')`). The unit tests for OBS-12
(`test_browser_role_prefix_hint.py`) pin the hint firing via
`generate_hints` in `utils/hints.py`.

The v3 validation run confirmed by direct test that **the hint never
actually fires in production**: Sonnet deliberately attempted
`intent_action(intent="click", target="button=Calculate")` on
Obstacle 8 and the OBS-12 hint was absent from the failure response.

Root cause: `_check_browser_role_prefix_misuse` is invoked via the
`generate_hints(ctx)` pipeline, which is called from the
**keyword-execution failure path** at
`keyword_executor.py:1684+`. But `button=X` against Browser library
typically fails earlier — at **pre-validation** (the 500ms gate that
checks element actionability) — which builds its own hint list
**inline** at `keyword_executor.py:1393+` and does NOT route through
`generate_hints`. The OBS-12 hint is therefore unreachable from the
path where the bug actually manifests.

The unit tests passed because they mocked the wrong call site. They
asserted the hint fires when `generate_hints(ctx)` is called with the
right HintContext — but they didn't verify that the production
failure path for `button=X` rejections actually calls
`generate_hints`. It doesn't.

### User story

> **As an** automation agent who accidentally uses a SeleniumLibrary-
> style role-prefix locator against the Browser library,
> **I want** the failure response to include the OBS-12 hint pointing
> at `text=` and `css=<tag>:text-is('X')` alternatives,
> **so that** I can fix the locator without spending another tool call
> guessing at the syntax. The current behaviour — generic "element
> not visible/enabled" with no syntax clue — is exactly what OBS-12
> was supposed to close.

### Details captured during the run

- **Obstacle**: 8 — Wait a moment (Sonnet v3 validation run)
- **Failing call**: `intent_action(intent="click", target="button=Calculate")`
- **Response error text**: `"Pre-validation failed: Element missing required states: enabled, visible"`
- **Hint surfaced**: visibility_hint, enabled_hint, pre_validate_timeout_hint
- **Hint MISSING**: `browser_role_prefix_misuse` (the OBS-12 hint)
- **Sonnet's verbatim observation**: *"Did the OBS-12 hint fire? **No.**
  The error returned was a pre-validation failure: 'Pre-validation
  failed: Element missing required states: enabled, visible'."*
- **Recovery cost**: 1 call (Sonnet retried with `text=Calculate`),
  effectively the same as if OBS-12 didn't exist.

### Acceptance criteria

1. When pre-validation rejects a locator that matches the OBS-12
   role-prefix pattern (`(button|link|input|select|textarea)=<value>`)
   in a Browser-library session, the failure response's `hints` list
   includes a `browser_role_prefix_misuse` entry with the same shape
   as the keyword-execution-path version (title, message, two
   examples for `text=` and `css=<tag>:text-is('X')`).
2. The existing OBS-12 unit tests must continue passing — the
   keyword-execution-path firing is preserved (it's still the
   fallback when pre-validation is disabled or doesn't run).
3. The hint MUST NOT fire twice when both paths could reach it
   (deduplicate by `type` key in the hints list).
4. Real-Browser integration test in
   `test_real_browser_prevalidation.py`: launch a Browser session,
   navigate to a fixture page with a `<button>Submit</button>`,
   call `intent_action(intent="click", target="button=Submit")`,
   assert the response's `hints` contains an entry with
   `type == "browser_role_prefix_misuse"` (or equivalent matcher
   based on the title text).

### Suggested experiments

- **Repro test (must fail before the fix)**: spin up a Browser
  session against a data: URL fixture page with one button. Call
  `intent_action(intent="click", target="button=Submit")`. Assert
  the response includes the role-prefix hint in `hints`. This test
  should FAIL on the current code (proving the gap) and PASS after
  the fix.
- **Re-run benchmark**: after OBS-15 lands, run Sonnet's v3 prompt
  again. Confirm the agent receives the hint when it tries
  `button=X` (the exact scenario from Obstacle 8).

### Implementation notes

- The pre-validation failure-hint block lives at
  `src/robotmcp/components/execution/keyword_executor.py:1393+`. It
  currently builds hints inline:
  - `pre_validation_failure` (always)
  - `visibility_hint` (if missing "visible")
  - `enabled_hint` (if missing "enabled")
  - `pre_validate_timeout_hint` (always)
- Add a check that calls
  `_check_browser_role_prefix_misuse(ctx_like, err)` (or equivalent
  inline duplicate) and appends to `hints` when the locator matches
  the role-prefix pattern.
- Build a small adapter to construct a `HintContext`-shaped object
  from the pre-validation block's available data (keyword, arguments,
  session library, error_text). The existing OBS-12 checker doesn't
  need to change — only the call site does.
- Alternative: extract the role-prefix-check logic into a shared
  helper in `utils/hints.py` that both paths import. Cleaner long-
  term but slightly larger refactor.
- The two paths emit hints with different schemas: the
  pre-validation path uses inline `{"type": "X", "message": ...,
  "suggestion": ..., "example": ...}` shapes; `generate_hints`
  returns `{"title": ..., "message": ..., "examples": [...]}` shapes.
  Reconcile the shapes in the pre-validation path so the OBS-15
  addition matches the existing pre-validation hint format.

### Out of scope

- Other locator-shape hints that may have the same reachability
  issue. Audit them in a follow-up if needed; OBS-15 just covers
  the role-prefix case.
- Rewriting the hint pipeline so all hints go through a single
  surface. Worth doing eventually but not blocking OBS-15.

---

## OBS-16 — `untracked_variables` warning false-positive after OBS-11 substitution

**Type**: Bug (low-impact) · **Priority**: Low · **Effort**: S

### Background

The v3 validation run surfaced a low-impact-but-confusing false
positive in `build_test_suite`'s `warnings` field. Sonnet's
generated suite for Obstacles 3 and 7 correctly used `${ORDER_ID}`
and `${ENTRY_COUNT}` in subsequent `Fill Text` steps (OBS-11
substitution working as designed). But the same suite's `warnings`
list included both variables in `untracked_variables` — declaring
them as captured-but-unused.

The fact that the variables DO appear in the rf_text contradicts
the warning. The likely root cause: the variable-tracking pass that
emits `untracked_variables` runs BEFORE OBS-11's
`_propagate_assigned_variables_to_literal_args` rewrites the
literals. The tracker sees only the literal `5` / `1079954` —
correctly identifies them as not-being-references to a captured
variable — and flags `${ENTRY_COUNT}` / `${ORDER_ID}` as untracked.

This is a fidelity issue, not a correctness issue. The generated
suite is correct; only the diagnostics lie. Low impact but worth
fixing because warnings exist precisely to inform agents about
real issues — false positives erode trust.

### User story

> **As an** agent reading `build_test_suite` warnings to confirm my
> recorded session is clean,
> **I want** the `untracked_variables` warning to reflect the
> post-substitution state of the generated suite,
> **so that** I don't have to verify whether each warning is real or
> an artifact of OBS-11's post-processing pass.

### Details captured during the run

- **Obstacles affected**: 3 (Not a table) and 7 (And counting),
  Sonnet v3 validation run
- **Variables flagged**: `${ENTRY_COUNT}` and `${ORDER_ID}`
- **Reality**: both variables ARE referenced in the rf_text after
  OBS-11 propagation (`Fill Text id=offerId ${ORDER_ID}`,
  `Fill Text ... ${ENTRY_COUNT}`)
- **Sonnet's verbatim observation**: *"`ENTRY_COUNT` and `ORDER_ID`
  appeared in `build_test_suite` `warnings[].untracked_variables`
  even though both were correctly assigned and propagated in the
  rf_text. The warning appears to be a false positive — the
  variables ARE tracked (they show up in the rf_text correctly)."*

### Acceptance criteria

1. When OBS-11 substitutes a literal arg with `${VAR}` in the
   generated suite, the resulting suite's `untracked_variables`
   warning MUST NOT include `${VAR}`.
2. Variables that ARE genuinely declared-but-unused (no captured
   value used downstream by any subsequent step) DO still appear
   in `untracked_variables` — no regression on the warning's
   intended purpose.
3. The fix is verified by a unit test that:
   a. Synthesizes a session with a capture step and a subsequent
      literal-arg step that matches the captured value.
   b. Builds the suite.
   c. Asserts the rf_text contains `${VAR}` AND the warnings list
      does NOT contain `${VAR}` in `untracked_variables`.
4. Negative test: a session with a capture step but no downstream
   reference (matching or otherwise) still produces the
   `untracked_variables` warning for that variable.

### Suggested experiments

- **Inspect ordering**: find the variable-tracking pass in the
  suite-build pipeline. Confirm it runs before OBS-11's
  `_propagate_assigned_variables_to_literal_args` (called from
  `_generate_rf_text`).
- **Two fixes possible**:
  a. Move the variable-tracker AFTER OBS-11 substitution (the
     tracker sees the post-substitution rf_text).
  b. Make the tracker re-scan after substitution.

### Implementation notes

- The OBS-11 substitution lives in `test_builder.py`
  `_propagate_assigned_variables_to_literal_args` and is called
  from `_generate_rf_text` per-test-case.
- The `untracked_variables` warning is constructed elsewhere —
  likely in a different pass that walks `session.steps` /
  `test_case.steps` BEFORE the rendering pass. Find it by grep:
  `grep -rn "untracked_variables" src/robotmcp/`.
- Simplest fix: ensure the OBS-11 substitution mutates
  `test_case.steps` in place (which it does), then re-run the
  variable-tracker after. Or move the tracker to run as part of
  the rendering pass so it sees the substituted args.

### Out of scope

- Other false-positive warnings in the suite-build pipeline. Audit
  separately if more surface.

---

## OBS-17 — ARIA snapshot conflates visible page text with element metadata

**Type**: UX · **Priority**: Low · **Effort**: S (heuristic-based)

### Background

The Tricentis Obstacle Course's "Not a table" page (Obstacle 3) has
a visible teaching annotation that reads, in part, *"ControlProperties:
class myBad..."* — as part of the page's RENDERED CONTENT, not as
real element class/id attributes. When this content is captured into
the ARIA snapshot, an agent reading the snapshot sees the phrase
`class myBad` and reasonably pattern-matches it as a CSS-class hint
to target: `css=.myBad`.

The locator doesn't match anything because no element has class
`myBad` — the text just happens to MENTION it. Sonnet wasted two
tool calls in v3 on this pattern.

This is a Tricentis-page-specific quirk, but the underlying issue is
general: when a page contains visible prose that contains words like
"class X" or "id Y", those phrases can mislead pattern-matching
agents. The ARIA snapshot could disambiguate by wrapping visible-text
content in clearly-delimited quotes/brackets, so the agent can
distinguish:
- `<div class="X">` (real element class)
- visible text "class X" (page prose)

### User story

> **As an** agent reading an ARIA snapshot to derive locators,
> **I want** visible-text content rendered in a way that clearly
> distinguishes it from element-metadata fields (class names, ids,
> roles),
> **so that** I don't pattern-match prose like "class myBad" as a
> CSS-class hint when it's actually just visible page text.

### Details captured during the run

- **Obstacle**: 3 — Not a table (Sonnet v3 validation run)
- **Sonnet's verbatim observation**: *"ARIA text showed 'class myBad'
  as part of ControlProperties metadata text, leading me to try
  `.myBad` as CSS class — wasted 2 tool calls. The metadata is
  embedded in the page text, not real element attributes."*
- **Wasted call 1**: `Get Text css=.myBad td:last-child` — failed,
  `css=.myBad` matched nothing
- **Wasted call 2**: `Evaluate JavaScript` targeting `.myBad` —
  same root cause
- **Working recovery**: read the FULL HTML via `Get Page Source` to
  discover the actual `.propertyGrid` CSS class on the real DOM
  element

### Acceptance criteria

1. When the ARIA snapshot renders an element's `visible-text`
   content (the text actually displayed in the browser), the text
   is wrapped in distinguishing characters (e.g. surrounding
   double-quotes `"..."` or `[text: ...]`) so it's clearly
   distinguishable from element-attribute fields.
2. The wrapping is consistent across all visible-text-bearing
   elements; agents can scan for the wrapper to find prose.
3. Element-metadata fields (`role`, `name`, `accessibility name`,
   etc.) remain rendered in their existing format — only
   `visible-text` / `text-content` is wrapped.
4. Existing ARIA-snapshot integration tests continue passing (the
   wrapping is additive, doesn't break parsers).
5. Pin via unit test: a fixture page with a `<div class="real">prose
   that mentions class myBad and id foo</div>` — the ARIA snapshot
   should clearly distinguish the real `class="real"` attribute
   from the visible text containing "class myBad".

### Suggested experiments

- **Repro fixture**:
  ```html
  <div class="real-class">This div mentions class fakeBad as part of its prose.</div>
  ```
- ARIA snapshot output should make it clear that `real-class` is
  the element's class (metadata) and `class fakeBad as part of its
  prose` is the rendered text content.
- **A/B**: re-run Sonnet's v3 prompt after OBS-17 lands. Confirm
  the wasted-call pattern on Obstacle 3 doesn't reproduce.

### Implementation notes

- ARIA snapshot generation lives in
  `src/robotmcp/components/execution/page_source_service.py`. Find
  the text-rendering path (likely a method that formats each
  element's role + name + value + text).
- The natural fix is to wrap visible text in double quotes when
  emitting:
  - Before: `div ControlProperties: class myBad ...`
  - After: `div "ControlProperties: class myBad ..."`
- Some MCP-server frameworks already use this convention; Playwright's
  ARIA snapshot output style is a reasonable reference.

### Out of scope

- Page-source filtering / sanitisation. Just the ARIA-snapshot
  text-rendering aspect.
- ARIA snapshot semantics (which elements are visible, role
  inheritance, etc.) — unchanged.

---

## Open investigations (not yet formal stories)

### IH1 — Haiku's OBS-11 propagation rate was 1 of 3

Haiku's v3 report listed three captured variables:
- `${TOTAL_COUNT}` — propagated correctly
- `${ORDER_ID}` — used as literal (not propagated)
- `${ENTRY_COUNT}` — used as literal (not propagated)

Three plausible causes for the misses (cannot disambiguate without
the raw Haiku rf_text):

1. **Lookback expired**: Haiku may have inserted >10 intermediate
   steps between capture and use, exceeding OBS-11's lookback window.
2. **Locator-slot rule**: Haiku may have placed the literal at
   arg 0 (the locator position), where OBS-11 deliberately skips
   substitution.
3. **Type-coerce mismatch**: `Get Element Count` returns int at
   runtime; `_captured_value_str` coerces to str. The literal in
   the next step should also be a string (RF wire format). Possible
   edge case if Haiku used a numeric form somewhere.

**Action**: before promoting this to a story, retrieve Haiku's raw
rf_text from the v3 transcript and inspect the actual capture/use
patterns to determine whether OBS-11's bounding rules are too tight
or whether Haiku's calls fall outside the rules legitimately. If the
former: file as OBS-18 (relax lookback / arg-0 rules) with
acceptance criteria measurable against the same Haiku-style call
trace.

---

## Roll-up: experiments to run after this batch

A single re-run of the v3 benchmark after OBS-15/16/17 ship should
confirm:

- **OBS-15**: Sonnet's `button=Calculate` attempt on Obstacle 8
  yields the role-prefix hint in the response. Reduces the recovery
  cost from 1 call to 0 (agent reads the hint instead of guessing).
- **OBS-16**: `build_test_suite` warnings list does not include
  `${ENTRY_COUNT}` or `${ORDER_ID}` (they're correctly tracked
  post-substitution).
- **OBS-17**: Sonnet's wasted-call pattern on Obstacle 3 (`class myBad`)
  does not reproduce. The ARIA snapshot makes visible-text vs
  metadata distinguishable.

## Related artifacts

- `docs/benchmarks/2026-05-17_tricentis_v3_post_obs_10_11_12.md` —
  parent v3 benchmark report.
- `docs/issues/2026-05-17_post_obs_followup_stories.md` — predecessor
  stories OBS-10/11/12 (now landed on `feature/obstacle-course-followup`)
  + OBS-13/14 still proposed.
- `docs/issues/2026-05-17_obstacle_course_stories.md` — original
  OBS-01..09 series (now merged via PR #69).
