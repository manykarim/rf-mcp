# Tasks: desktop-aut-process-lineage

## 1. Lineage check

- [x] 1.1 Add `pid_in_aut_lineage(target_pid, aut_pid, aut_sid, *, _ppid=None, _sid=None) -> Optional[bool]` to `platynui_focus.py`: pid equality → ancestor walk (≤15 hops via `/proc/<pid>/status` PPid, stop at 0/1) → session-id match → False only when both signals resolved negative; None on any read failure; injectable readers
- [x] 1.2 Capture `desktop_aut_sid` (`os.getsid(pid)`, best-effort) in the executor launch block next to `desktop_aut_pid`; declare the field on `ExecutionSession`
- [x] 1.3 Rewrite the `ensure_focused` aut check: accept `aut_sid` kwarg, call `pid_in_aut_lineage`, warn ONLY on `False` with the lineage-stating message (pids + session ids); `_platynui_focus_before_act` forwards `aut_sid`
- [x] 1.4 Update the `test_focus_verifiability.py` PID-mismatch tests to the lineage contract (foreign = no relation on ANY tier)

## 2. Tests

- [x] 2.1 `tests/unit/test_aut_process_lineage.py` with fake `_ppid`/`_sid` trees: wrapper child in scope; daemonized-reparented-same-sid in scope; single-instance handoff in scope; foreign pid+sid warns with both ids in the message; dead-launcher indeterminate silent; hop limit respected
- [x] 2.2 Live smoke (Linux): launch `bash -c 'sleep 5 & wait'` via the engine's launch path, resolve the sleep child's pid, assert `pid_in_aut_lineage` True via real /proc readers

## 3. Validation

- [x] 3.1 Full unit suite green (baseline 6805 passed + 1 skipped; no regressions) — **6817 passed + 1 skipped, +12 net**
