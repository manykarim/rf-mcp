# Proposal: desktop-aut-process-lineage

## Why

Run 5 (2026-06-12) showed the AUT process-identity scope check (added in platynui-visible-safe-targeting) **false-positives on every interaction** when the agent launches the AUT through a wrapper: `Start Process /usr/bin/bash launch_libreoffice_writer.sh` captures the bash pid (992737) as `desktop_aut_pid`, while the resolved target's `app:Application` reports the real soffice pid (992769) — so every click warned "commands may be going to a different application" against the correct target. Two structural twists make a naive parent-child fix insufficient: LibreOffice daemonizes (oosplash forks and exits, soffice re-parents to init — ancestor walks dead-end), and its single-instance handoff means a reopen launch (new launcher pid 996321) is served by the **original** soffice (992769). A warning that fires on every legitimate interaction trains agents to ignore it.

## What Changes

- The scope check compares **process lineage**, not bare pid equality. A resolved target is in scope when ANY of: target pid == launched pid; the target's `/proc` parent chain reaches the launched pid (wrapper children); or the target's **session id** matches the session id captured at launch (`os.getsid`) — which survives both daemonization and single-instance handoff, because everything the MCP server spawns shares its session id unless something calls `setsid`.
- The launch block captures `desktop_aut_sid` alongside `desktop_aut_pid` (best-effort).
- **Fail-open on indeterminacy**: when the relationship cannot be established (launcher dead and `/proc` unreadable for either side), no warning is emitted — the warning's value is high-confidence "this is a foreign application", not lineage bookkeeping noise.
- The warning fires only on a CONFIRMED foreign process (both sides resolvable, no lineage relation).

No breaking changes — the warning becomes rarer and correct; response shapes unchanged.

## Capabilities

### New Capabilities

- `desktop-aut-process-lineage`: AUT scope verification by process lineage (pid identity, ancestor chain, session id) with fail-open indeterminacy, replacing bare pid equality.

### Modified Capabilities

(none archived match; the prior `desktop-focus-verifiability` spec's PID-mismatch scenario remains satisfied — a genuinely foreign pid still warns)

## Impact

- `src/robotmcp/components/execution/platynui_focus.py` — lineage helpers (`_pid_ancestors`, `_pid_sid`, `pid_in_aut_lineage`) + the `ensure_focused` aut_pid check rewritten to use them; `aut_sid` kwarg.
- `src/robotmcp/components/execution/keyword_executor.py` — launch block stores `desktop_aut_sid`; `_platynui_focus_before_act` passes it.
- `src/robotmcp/models/session_models.py` — `desktop_aut_sid: Optional[int]` field.
- Tests: `tests/unit/test_aut_process_lineage.py` with injectable `/proc` readers pinning the run-5 shapes (wrapper child, daemonized-same-sid, single-instance handoff, true foreigner, dead launcher); existing `test_focus_verifiability.py` PID tests updated for lineage semantics; baseline 6805 passed + 1 skipped stays green.
