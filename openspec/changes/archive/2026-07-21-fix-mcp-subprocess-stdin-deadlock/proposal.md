## Why

On Windows, `run_test_suite(mode="dry", suite_file_path=...)` hangs for the full 180 s
`DRY_RUN_TIMEOUT` and returns "Dry run execution timed out after 180s" — reproduced
"multiple times" on a real client. A full Windows diagnostic (evidence in
`experiments/dryrun_windows_timeout_results.md`) proved the cause is **NOT** the suite or
PlatynUI (a trivial `Log hi` suite hangs identically):

- The stdio MCP server holds a permanent **pending blocking read on its stdin pipe**
  (FastMCP JSON-RPC transport).
- The dry-run helper spawns `subprocess.run([python, -m, robot, --dryrun], capture_output=True)`
  which redirects stdout/stderr **but not stdin**, so the child **inherits the server's
  stdin pipe handle**.
- On Windows a *synchronous* file handle serializes operations, so the child's C-runtime
  stdio init (`fflush → _lseeki64 → SetFilePointerEx → NtQueryInformationFile`, confirmed
  by a py-spy native stack with no app frames) **blocks behind the parent's pending read**
  → deadlock until the timeout. POSIX pipes do not serialize this way (Linux = ~0.9 s).

A deterministic repro confirmed it (pending-read stdin → hang) and confirmed the fix
(`stdin=subprocess.DEVNULL` → OK). A second finding: on timeout, `subprocess.run` cannot
reap the child cleanly because killing the uv-trampoline launcher orphans the real
interpreter, which holds the capture pipes open — so `communicate()` itself hangs.

The same inherited-stdin pattern exists in a second, latent site
(`library_checker` `pip list`), so this is a **server-wide subprocess-hygiene** defect,
not a one-off.

## What Changes

- **No serving-time subprocess inherits the server's stdin.** Pass
  `stdin=subprocess.DEVNULL` to the dry-run subprocess (`suite_execution_service.py`
  `run_robot_dry`) — the primary, validated fix — and to the latent `library_checker`
  `pip list` spawn and the Linux-only X11 probes in `platynui_focus.py` (harmless there,
  consistent policy).
- **Timeouts reap the whole process tree.** On dry-run timeout, kill the child *and its
  descendants* (Windows `taskkill /F /T /PID`; POSIX `killpg`/process group) so a hung
  `robot` (and the uv-trampoline → real-interpreter chain) does not orphan or block the
  post-timeout `communicate()`.
- **Defense-in-depth:** add `--console none` to the dry-run options (mirroring the
  in-process normal-run path) so RF console writes cannot reach an inherited handle.
- **Regression coverage:** a Windows-representative unit test that fails if a serving-time
  subprocess is spawned without an explicit non-inheriting stdin.

## Capabilities

### Modified Capabilities
- `mcp-stdio-log-safety`: extends the stdio-safety invariants to child processes — the
  server MUST NOT let spawned subprocesses inherit its stdin (the JSON-RPC read channel),
  and subprocess timeouts MUST reap the whole process tree.

## Impact

- **Code:** `src/robotmcp/components/execution/suite_execution_service.py` (`run_robot_dry`
  stdin + process-tree kill on timeout + optional `--console none`),
  `src/robotmcp/utils/library_checker.py` (pip-list stdin),
  `src/robotmcp/components/execution/platynui_focus.py` (X11 probe stdin, consistency).
- **Behaviour:** Windows dry-run goes from a 180 s hang to ~1 s; the latent `pip list`
  10 s Windows hang is closed. No Linux behaviour change (validated: dry-run with
  `stdin=DEVNULL` = 0.48 s). The full-run path is unaffected (it uses in-process
  `run_cli`, not a subprocess).
- **Non-goals:** changing the dry-run to run in-process (kept as a subprocess for RF
  console-output isolation); the PlatynUI stack (fully exonerated).
