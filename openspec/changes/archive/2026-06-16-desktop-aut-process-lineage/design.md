# Design: desktop-aut-process-lineage

## Context

Run-5 evidence: every interaction carried *"resolved target belongs to PID 992769, but the AUT was launched as PID 992737"* — bash wrapper (992737) → oosplash → soffice.bin (992769, re-parented to init after oosplash exited); the reopen launch (996321) was handed off to the same soffice 992769 (LibreOffice single-instance). The current check (`ensure_focused`, platynui_focus.py) is `int(target_pid) != int(aut_pid)` → warn.

Verified on this host: `os.getsid` children inherit the parent's session id; nothing in the Start Process → bash → libreoffice chain calls `setsid`, so the daemonized soffice retains the MCP server's session id. SID is therefore the lineage signal that survives both daemonization and handoff.

## Goals / Non-Goals

**Goals:** zero false positives on the three run-5 shapes (wrapper child, daemonized AUT, single-instance handoff); a genuinely foreign application (different session — e.g. a host app leaking into the tree) still warns; indeterminate cases stay silent.

**Non-Goals:** Windows lineage (the check already only runs where `/proc` semantics exist — gate on platform); tracking multiple concurrent AUTs beyond the shared-session boundary; replacing the display-scoping work (D4 of evidence-and-display-scoping) — lineage refines the *warning*, scoping governs *discovery*.

## Decisions

### D1 — Three-tier lineage check, injectable for tests
`pid_in_aut_lineage(target_pid, aut_pid, aut_sid, *, _ppid=…, _sid=…) -> Optional[bool]` in platynui_focus.py:
1. `target == aut` → True.
2. Ancestor walk: follow `/proc/<pid>/status` PPid from target, ≤15 hops, stop at 0/1; chain contains aut → True (covers live wrapper parents).
3. SID: `os.getsid(target) == aut_sid` → True (covers daemonization + handoff; everything the server session spawned shares it).
4. Both signals resolvable and negative → False (warn). Any read failing (process gone, no /proc) → None (indeterminate, no warn).
Readers injectable (`_ppid(pid)->Optional[int]`, `_sid(pid)->Optional[int]`) so tests run on fake process trees.

### D2 — Capture `desktop_aut_sid` at launch
The existing launch block (keyword_executor, where `desktop_aut_pid` is captured from `proc.pid`) also stores `os.getsid(pid)` best-effort. Declared on `ExecutionSession`. `_platynui_focus_before_act` forwards `aut_sid` to `ensure_focused`.

### D3 — Warning text gains the lineage verdict
On a confirmed foreign target the message stays substantively the same but states what was checked: *"resolved target (PID X, session S1) has no lineage relation to the launched AUT (PID Y, session S2) — commands may be going to a different application"*. Indeterminate → silent (fail-open), matching the codebase's warn-only-on-confidence convention.

### D4 — Platform gate
Ancestor/SID reads are Linux-/proc-based; on platforms where they raise, the readers return None → tier-4 indeterminacy → silent. No behavioral change off-Linux versus today's noisy check being mostly unreachable there anyway.

## Risks / Trade-offs

- [SID boundary is "everything this server spawned", wider than one AUT] → exactly the trust boundary the guard protects (server-launched processes vs. foreign desktop apps); two AUTs launched by the same session are both legitimate targets.
- [A host app could theoretically share the SID if the server were launched from the same terminal session as the user's apps] → the server runs under the MCP client (codex/IDE), a distinct session from desktop-launched GUIs (verified: session leaders differ); residual risk is acceptable for an advisory warning.
- [PID reuse between capture and check] → same races as before; advisory only.

## Migration Plan

Additive + warning-behavior refinement. Update the two `test_focus_verifiability.py` PID tests to the lineage contract deliberately. Rollback = revert.

## Open Questions

(none)
