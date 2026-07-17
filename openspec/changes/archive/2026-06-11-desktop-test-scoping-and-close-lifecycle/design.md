# Design: desktop-test-scoping-and-close-lifecycle

## Context

Run-3 findings with root causes (researched 2026-06-11):

| # | Symptom | Root cause |
|---|---------|------------|
| 1a | Early `start_test` → "No context for session" | server.py start_test handler (~3505-3567) never calls `create_context_for_session`; only `execute_step` creates the context |
| 1b | Failure swallowed; layers diverge | the context-layer error is logged as a warning and start_test returns success — registry activates multi-test mode while the RF context has no test |
| 1c | 43/46 steps in `suite_level_steps`; suite renders 3 steps | `ExecutionSession.add_step` routes to `suite_level_steps` when multi-test mode is on but `get_current_test()` is None; `build_test_suite` never iterates `suite_level_steps` — silently dropped |
| 1d | Final `end_test` → "No active test to end" | `end_test_in_context` (rf_native_context_manager.py:776) checks the CONTEXT layer's `current_res_test`, which never matched the registry |
| 3 | LibreOffice survived window close as a start-center frame, no signal | no post-close liveness check exists (launch has one — ADR-029 `launch_liveness_hint`) |
| 5 | Empty display reported "(X11 probe unavailable)" | `_build_exposure_diagnostic` calls `x11_window_present(app_names=None)` which returns "unknown" by contract when given nothing to match; the new `x11_display_pids()` can distinguish empty-and-reachable (`frozenset()`) from probe failure (`None`) |

## Goals / Non-Goals

**Goals:** a `start_test` that succeeds is consistent across BOTH layers (registry + RF context) regardless of call order; an `end_test` after a successful `start_test` always succeeds; steps recorded while a test is active land in that test; steps that end up suite-level are visible, never silently dropped; closing a desktop window that leaves the AUT alive produces a signal; an empty display is diagnosed as empty.

**Non-Goals:** auto-merging `suite_level_steps` into test bodies (ownership is ambiguous — visibility, not adoption); a CLOSE_APP intent verb (agents already compose Close Window + Terminate Process; the missing piece was the signal — a composite intent can ride a later change); detecting keyboard-driven closes (`<Alt+F4>`) — keyword-level `Close Window` only, the hint text covers the rest; fixing LibreOffice's start-center behavior.

## Decisions

### D1 — start_test creates the context and is atomic
The start_test handler, before `start_test_in_context`: if `get_session_context_info(session_id)["context_exists"]` is false, call `create_context_for_session(session_id, libraries=session.search_order or loaded_libraries)` — the same recipe `_execute_keyword_with_context` uses. If context creation OR `start_test_in_context` then fails, return `{success: False, error: …}` WITHOUT touching the TestRegistry — no half-activated multi-test mode. Order: context first, registry last (registry activation is the irreversible bit).
*Alternative rejected*: keeping the soft-warning behavior and only fixing the registry routing — a "successful" start_test that didn't start anything is the lie that cost run 3 its suite.

### D2 — end_test is registry-first
The end_test handler ends the registry test when one is active (source of truth for suite generation); the `end_test_in_context` result is attached as `context_result` and a context-layer miss ("No active test to end" from the RF layer) becomes a soft `warning` on a successful response instead of failing the call. When the REGISTRY has no active test either, the call still fails as today.
*Why registry-first*: build_test_suite renders from the registry; the context layer exists for RF-native variable scoping and is self-healing on the next start.

### D3 — suite-level steps become visible in build_test_suite
`build_suite` adds `suite_level_step_count` to the response and a top-level warning (composes with the I-1 empty-suite warning; both may appear via a list-or-priority — implementation: the more specific scoping warning wins when both fire) when `len(session.suite_level_steps) > total in-test steps` and the registry is in multi-test mode: *"N recorded step(s) sit OUTSIDE any named test (suite-level) and are not rendered in test bodies — start_test before executing steps, or rebuild without multi-test mode."*
*Alternative rejected*: rendering suite_level_steps as a synthetic test case — surprising output and wrong semantics for setup-ish steps.

### D4 — regression test reproduces the run-3 interleaving
The decisive test drives the REAL handlers: init session → start_test (no prior execute_step) → several execute_steps → build_test_suite (twice, interleaved — run 3 called it 10×) → more steps → end_test. Asserts: start_test succeeds, every step lands in the named test, suite body contains them, end_test succeeds. This pins whatever mechanism diverted run-3's steps (the interleaved build path is suspected of contributing via the context-recreation path already fixed in D6 of the previous change).

### D7 — build_test_suite no longer auto-ends the running test (discovered during apply)
The D4 regression test exposed the ACTUAL mechanism behind run 3's 43 orphaned steps: `build_test_suite` in multi-test mode auto-ended any running test (test_builder.py:301-313, `end_test(status="pass")` + `end_test_in_context`). Stepwise agents call build between steps — the first build silently closed the active test, and every later step fell into `suite_level_steps`. Fix: the build is non-destructive; the still-running test's live step list renders without ending it, and recording continues into it afterwards. The legacy test pinning auto-end (`test_auto_end_running_test`) was rewritten to pin the new contract.

### D5 — close liveness hint
`desktop_execution_signals.py` gains `is_close_keyword()` (`close window` basename) and `close_liveness_hint(aut_pid_alive: Optional[bool])` returning a hint when True. The executor's post-success desktop block (next to the launch-liveness block) checks: close keyword + `session.desktop_aut_pid` set + `os.kill(pid, 0)` succeeds (subprocess-free liveness probe; `ProcessLookupError` → dead, `PermissionError` → alive) → hint: window closed, process alive, residual frame likely, name `Terminate Process` as the hard stop. Pure decision function + thin wiring, matching the module's contract.

### D6 — empty-display diagnostic via the batched PID probe
In `_build_exposure_diagnostic`, when `presence == "unknown"` AND no app_filters were given: call `x11_display_pids()` — `frozenset()` → new diagnostic `{type: "display_empty", window_present: False, message: "The display is reachable but has no application windows — the AUT has not been launched on this display yet."}`; non-empty set → fall through to the existing undetermined wording (windows exist but didn't resolve by name — the current message is then accurate); `None` → unchanged "(X11 probe unavailable)". Reuses the cached `_display_scoped_pids()` to avoid a second subprocess when scoping already probed.

## Risks / Trade-offs

- [D1 makes early start_test slower (context creation ~100-300ms)] → one-time per session; identical cost to what the first execute_step already pays.
- [D1 turns previously-"successful" calls into failures] → that success was false; the error now carries the real cause and the agent retries with signal. Existing tests asserting the soft-success shape must be updated deliberately.
- [D2 could mask genuine context-layer bugs] → the context_result with the miss is still in the response; only the success flag changes.
- [os.kill(pid,0) races with PID reuse] → window between close and check is milliseconds; worst case is a spurious advisory hint.
- [D3 warning could fire on legitimate suite-level setup steps] → threshold (suite-level > in-test) plus multi-test-mode gate keeps deliberate single-test and setup-light sessions silent.

## Migration Plan

Additive plus one loud-failure behavior change (D1) documented in the start_test docstring. Rollback = revert. Baseline 6774 passed + 1 skipped stays green.

## Open Questions

- Should `build_test_suite` later gain `adopt_suite_level="current_test"|"new_test"` to recover orphaned steps? Deferred until an agent actually needs recovery rather than prevention.
