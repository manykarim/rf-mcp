## Context

rf-mcp injects agent-facing guidance through two surfaces: MCP server `instructions`
(templates in `domains/instruction/value_objects.py`, selected by
`ROBOTMCP_INSTRUCTIONS[_TEMPLATE]`) and tool docstrings (`@mcp.tool` in `server.py`). An
instruction-quality investigation measured how these drive tool-calling.

Evidence (all live; success=succ, completion=comp, right-order=order, turns; glm-4.7-flash
unless noted):
- **N=5 tie-break (suite-exec, XML):** `minimal`(248ch) best-or-near-best on succ/comp/order
  with fewest turns (11); `standard`(2800ch) comp 0.40 suite / succ 0.48 XML (lowest);
  `detailed`(2700) comp 0.00 both, most turns (21–28); `discovery_first`(6081) middling +
  huge. `off` (docstrings only) comp 1.00 on suite but lost order (0.55) on hard XML.
- **Matrix (glm+qwen, generic+API):** on the hard API scenario, the shorter sets
  (`off`/`minimal`/`refactored`) reached `get_locator_guidance` 100% and scored highest
  (0.78–0.84); the verbose ones 0–50% and lower — a mechanistic link (short → finds the
  right tool → better API).
- **Qualitative review:** 3 competing "first tools"; `standard` self-contradicts;
  `analyze_scenario` (docstring: "FIRST call") absent from the default template;
  `manage_session(init)` injects `desktop_guidance` but nothing for API; `.txt` files dead
  and divergent; ~40% of long templates is docstring-echo.
- **Haiku (real Claude Code client):** rf-mcp connects once the project is trusted; Haiku
  churned on the `analyze_scenario`/`manage_session` dual entry (2× redundant
  `manage_session`); one explicit sentence removed the churn (→ 0). Loading all 6 MCP
  servers overflowed Haiku's prompt (rf-mcp's 19 verbose docstrings are heavy).
- **Haiku, post-cold-start-fix (2026-07-20, updated finding — supersedes "Haiku can't
  complete"):** the earlier "Haiku could not complete a trivial task" was NOT a capability
  floor — it was a stdio bug (commit a467c5a): RF's global console logger leaked the
  test-start banner to fd 1 on the first keyword of a cold server, corrupting JSON-RPC and
  hanging the client until timeout, on EVERY run regardless of instructions. With that
  fixed, Haiku completes the full flow via the Claude Code CLI (`analyze_scenario →
  execute_step… → build_test_suite → run_test_suite`, suite runs+passes, ~28s) steered
  ONLY by server instructions. An n=8 (then n=20) candidate sweep on Haiku (neutral prompt;
  candidates differ only in server `instructions`) shows **100% completion (5/5 built,
  5/5 `success`) for ALL candidates**. Firmed n=5 ranking (avg session churn / avg calls /
  avg dur): **`checklist` 0.0 / 4.4 / 22.2s (WINNER — 0 churn every run, fewest calls,
  fastest) ≈ `example` (worked-example) 0.0 / 5.0 / 27.6s > `minimal` 0.2 / 5.0 / 30.8s >
  `terse` 0.4 / 4.4 / 26.0s**. (The n=2 pilot's "terse 3.0 churn" was a single-run outlier
  that n=5 washed out — churn is low across the board once the unified-entry line is
  present; `checklist`/`example` are the only two at a clean 0.0.) The load-bearing line
  that keeps churn at 0 is the explicit unified-session-entry: "analyze_scenario CREATES
  the session and returns session_id; NEVER call manage_session(action=init) — the session
  already exists." Length is NOT the driver (checklist 800 ch ties/beats minimal 344 ch);
  EXPLICITNESS of the one order+entry rule is.

## Goals / Non-Goals

**Goals:**
- Make the default instructions a lean, order-explicit spine aligned to the docstrings.
- One canonical first tool (`analyze_scenario`) + unified session entry (kill the churn).
- Guaranteed init-response guidance for instruction-sensitive libraries (API mirrors desktop).
- Tighten the dense docstrings (they drive behaviour AND bloat small-model context).
- Retire the verbose/dead templates.
- Every behaviour-affecting change validated by the `agentic-e2e-instruction-quality` gate.

**Non-Goals:**
- Changing tool names/signatures.
- Per-context auto-selection of templates (stays an env var).
- Making Haiku (small) reliably drive rf-mcp — the target is medium models.
- Rewriting the domain templates (`desktop_focused` is already the value-per-char model).

## Decisions

1. **Lean order-explicit default (~250–400 ch), aligned to docstrings.** Replace
   `standard` as default with a spine naming `analyze_scenario` first and the canonical
   order, plus "never guess — discover" and a one-line pointer to `get_locator_guidance`.
   *Rationale:* `minimal` empirically won and `off` was competitive; the 2800-char default
   was among the worst. *Alternative:* keep `standard`, just fix its self-conflict —
   rejected: even a consistent verbose template (`detailed`) had the worst completion.
   *Note:* the earlier "keep docstrings as-is" caution is superseded for THIS change (its
   explicit purpose is to refactor the instruction surface), but the caution is honored by
   gating every change on the e2e instruction-quality gate.
   *Concrete draft (the Haiku-sweep `checklist` winner — starting point for apply, tune
   against the reference-model A/B):*
   ```
   rf-mcp checklist, follow in order:
   1. Call analyze_scenario ONCE. It CREATES the session and returns session_id. Reuse
      that session_id in EVERY later call. NEVER call manage_session(action=init) — the
      session already exists.
   2. If a keyword or library is unknown, call find_keywords first.
   3. Add ONE keyword per execute_step(keyword, arguments=[...strings], session_id).
      arguments is always a list of strings.
   4. If execute_step fails: read the error, fix the keyword name/args via find_keywords,
      retry ONCE. Do NOT repeat the same failing call.
   5. Locators/API: get_locator_guidance(library="requests"|"browser"|...). Web DOM:
      get_session_state.
   6. FINAL: once steps pass, call build_test_suite(test_name, output_path). The task is
      NOT done until build_test_suite succeeds.
   ```

2. **Unify session entry on `analyze_scenario`.** Template + `analyze_scenario` docstring
   say it creates the session and is the front door; `manage_session(init)` docstring
   points to `analyze_scenario` for a new scenario. *Rationale:* the dual entry caused
   measured churn (Haiku 2× `manage_session`; also seen as glm `standard` session-churn).

3. **Guaranteed init-guidance injection (mirror desktop).** Add `api_guidance` to
   `manage_session(init)` when RequestsLibrary is present, reusing
   `utils/requests_guidance.py`. *Rationale:* the reachability defect — desktop gets
   guaranteed delivery, API does not; API is the turn-sink. *Alternative:* a template
   pointer to `get_locator_guidance(requests)` — weaker (agent must choose to call it;
   evidence shows shorter templates get there ~100% but it's still indirect); init
   injection is guaranteed.

4. **Docstring tightening, gate-validated.** Lead with when-to-call + primary modes/params;
   move secondary mechanics (OBS-19/21 externalisation, `record`/`pre_validate_timeout_ms`
   internals) to one line. *Rationale:* docstrings are the real driver and bloat context;
   the e2e gate catches regressions.

5. **Delete dead `.txt` templates.** They aren't loaded and disagree with production. Or
   wire the loader to read them as the single source — deleting is simpler and lower-risk.

## Risks / Trade-offs

- **Docstring/template change silently degrades a model not on the gate** →
  Mitigation: the reference-model gate + the (weekly) roster; recapture the reference
  baseline as a reviewed diff after the intended change; the change is the "no decrease"
  ratchet's whole point.
- **Removing content the verbose templates carried (recovery ladder, BDD rules)** →
  Mitigation: keep the genuinely non-discoverable one-liners in the lean spine or move
  them to `get_locator_guidance`/init-injection; the recovery ladder was already drowned
  in `discovery_first` so few agents used it.
- **API init-injection increases the init response size** → Mitigation: attach a compact
  rule set + pointer, not the full cookbook; measure token cost; it replaces many wasted
  Evaluate turns (net win, per the F-API1 history).
- **Baseline for the reference is sparse (only `minimax_basic_list` currently validated)**
  → Mitigation: this change should recapture the reference baseline AND (ideally) validate
  at least one more scenario before/after, so the refactor's effect is measured, not
  assumed.

## Migration Plan

1. Add the lean default template alongside the existing ones (opt-in via env first).
2. Capture the reference baseline under the current default; then under the lean default;
   compare on the validated scenario set (suite-exec is the reliable one; add API once its
   completion definition + init-injection land).
3. Add the `api_guidance` init injection; re-measure the API scenario (expect fewer turns,
   higher first-try `get_locator_guidance`/response-access correctness).
4. Tighten the three dense docstrings; re-run the gate; recapture baseline.
5. Flip the default to the lean spine; delete the dead `.txt` files.
6. Rollback = revert the default-template selection (the old templates remain available).

## Open Questions

- ~~Exact lean-spine wording~~ **RESOLVED (2026-07-20 Haiku CLI sweep).** Adopt the
  **`checklist` spine** shape — a short numbered list, ~500–800 ch — because it was the
  empirically cleanest (n=5 on Haiku: 0.0 avg churn, fewest avg calls 4.4, fastest 22.2s,
  every run perfectly ordered), with `example` (worked-example) a near tie. Non-negotiable load-bearing
  line is the explicit unified-session-entry ("analyze_scenario CREATES the session and
  returns session_id; NEVER call manage_session(action=init) — the session already
  exists"). This RE-CALIBRATES the earlier "~250–400 ch, closer to `minimal`" target: on
  the CLI/Haiku axis, explicitness of the order+entry rule beat raw brevity (checklist
  800 ch > minimal 344 ch on churn). Keep the reference-model (glm/qwen) A/B during apply
  as the second axis — the winner should satisfy BOTH (short AND order/entry-explicit);
  the concrete draft is the `checklist` text (Impact §).
- Keep `discovery_first`/`detailed` as opt-in domain/edge templates, or delete entirely?
- Should the API init-injection be gated by an env flag (like other guidance), or on by
  default when RequestsLibrary is present?
