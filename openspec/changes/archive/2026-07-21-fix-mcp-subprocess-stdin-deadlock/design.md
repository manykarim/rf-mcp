## Context

rf-mcp runs as a stdio MCP server: FastMCP reads JSON-RPC requests from `stdin` and writes
responses to `stdout` continuously, so the process **always has a pending blocking read on
its stdin handle**. Several tools spawn child processes with `subprocess.run(...,
capture_output=True)`, which redirects stdout/stderr to fresh pipes but **leaves stdin
inherited** from the parent.

The dry-run path (`suite_execution_service._execute_rf_dry_run` → `run_robot_dry`) spawns
`python -m robot --dryrun <suite>` this way. On Windows this deadlocks; on Linux it does
not. Full Windows evidence: `experiments/dryrun_windows_timeout_results.md` (and the
playbook `experiments/dryrun_windows_timeout_diagnostics.md`).

## Root cause (proven on Windows, reproduced deterministically)

- The child inherits the parent's **stdin pipe handle**. Windows opens inherited pipe
  handles as **synchronous** file objects, which serialize all operations on the handle.
- During interpreter/C-runtime startup the child performs stdio setup on its std handles
  (`fflush → _lseeki64 → SetFilePointerEx → NtQueryInformationFile`). On the shared
  synchronous stdin handle, that operation **blocks behind the parent's outstanding
  blocking read** — a cross-process deadlock. It only releases when rf-mcp's 180 s timeout
  fires.
- Confirmed by a py-spy **native** stack of the hung child: idle Python thread (no Python
  frames), native leaf `NtQueryInformationFile ← SetFilePointerEx ← fflush_nolock ←
  _lseeki64` — **no PlatynUI / robot / import frames**. Deterministic repro: a parent
  thread holding a pending `os.read` on a synchronous pipe + a child inheriting it as stdin
  → hang; `stdin=DEVNULL` → OK; no pending read → OK.
- **Suite-agnostic** (`Log hi` hangs too) → PlatynUI, Browser, and the RF suite content are
  all exonerated. POSIX pipes are not serialized this way, so Linux is unaffected (~0.9 s).

## Goals / Non-Goals

**Goals**
- Windows dry-run completes in ~1 s instead of a 180 s timeout, with no Linux regression.
- Make "no serving-time subprocess inherits the server's stdin" an enforced invariant.
- Make dry-run timeouts actually reap the child tree (they currently cannot on Windows).

**Non-Goals**
- Moving dry-run in-process (the subprocess exists to isolate RF console output that
  `run_cli` can leak to fd 1; keep it).
- Any change to the full-run path (it already runs `run_cli` in-process — no child, no bug).
- PlatynUI changes (exonerated).

## Decisions

1. **`stdin=subprocess.DEVNULL` on every serving-time subprocess.** *Rationale:* removes
   the inherited-handle deadlock at the source; validated by the Windows repro (trial 3)
   and harmless on Linux (dry-run 0.48 s). `--dryrun` never reads stdin, so DEVNULL is
   safe. *Alternatives rejected:* `stdin=PIPE` then close (repro §7 D still hangs while a
   pending read exists — closing the child's write end doesn't detach the inherited
   parent handle); `CREATE_NO_WINDOW`/`DETACHED_PROCESS` (repro §7 E/F — no effect);
   file-redirect of stdout/stderr (fixes the *capture* pipe class but not the *stdin*
   inheritance).

2. **Process-tree kill on timeout.** *Rationale:* the diagnostic showed that on timeout,
   `subprocess.run` kills only the direct child; with the uv-trampoline launcher the real
   interpreter is orphaned and keeps the capture pipes open, so the post-kill
   `communicate()` hangs. Kill the whole tree: Windows `taskkill /F /T /PID`; POSIX start
   the child in its own process group (`start_new_session=True`) and `killpg` on timeout.
   This also cleans up any leaked descendant across the "multiple times" reports.

3. **`--console none` for dry-run (defense-in-depth).** Mirror the normal-run path so RF
   never writes console output toward an inherited/std handle. Secondary to #1.

4. **Fix the latent second site now.** `library_checker.check_pip_package_installed`
   (`pip list`, `capture_output`, no stdin, 10 s timeout) is the same defect with a shorter
   fuse; fix it in the same pass. The `platynui_focus.py` X11 probes are Linux-only
   (`DISPLAY`-gated) so cannot hit this on Windows, but get `stdin=DEVNULL` for a single
   consistent policy.

## Risks / Trade-offs

- **A future subprocess reintroduces the bug** → Mitigation: a small shared helper (e.g.
  `run_subprocess_isolated(...)`) that always sets `stdin=DEVNULL`, plus a unit test that
  asserts the dry-run/library-check spawns pass a non-inheriting stdin.
- **`taskkill`/`killpg` edge cases** (missing `taskkill`, permissions) → Mitigation:
  best-effort with fallback to the existing `kill()`; never raise from the reaper.
- **Cannot run the real hang in Linux CI** (it's Windows-only) → Mitigation: unit test the
  *call shape* (stdin argument present + non-inheriting) and the reaper, not the OS-level
  deadlock; keep the Windows evidence doc as the acceptance reference.

## Migration Plan

1. Add `stdin=subprocess.DEVNULL` to the dry-run subprocess; re-run the Windows repro
   (expect ~1 s) and the Linux suite (expect no change).
2. Add the process-tree reaper on the dry-run timeout path.
3. Apply `stdin=DEVNULL` to `library_checker` (+ X11 probes for consistency); optionally add
   `--console none` to dry-run.
4. Add the regression unit test(s). Rollback = revert the stdin/kwarg additions (pure
   additive change).

## Open Questions

- Introduce the shared `run_subprocess_isolated` helper now, or inline `stdin=DEVNULL` at
  each site and add the helper later? (Leaning helper — it's the durable guard.)
- Should the dry-run default timeout drop from 180 s (it only ever mattered because of this
  deadlock; a real dry-run is ~1 s)? Out of scope here, worth a follow-up.
