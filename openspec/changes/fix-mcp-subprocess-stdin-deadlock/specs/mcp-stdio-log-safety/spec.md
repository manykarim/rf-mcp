## ADDED Requirements

### Requirement: Spawned subprocesses never inherit the MCP server's stdin

Every subprocess spawned by the stdio MCP server during request serving SHALL be given an
explicit non-inheriting stdin (e.g. `stdin=subprocess.DEVNULL`), and MUST NOT inherit the
server's own stdin handle, because that handle carries the JSON-RPC request stream and
always has a pending blocking read. On Windows an inherited stdin pipe is a synchronous
handle whose C-runtime stdio operations in the child deadlock behind the parent's pending
read (proven: a trivial dry-run hangs the full 180s timeout, native stack idle in
`NtQueryInformationFile ← SetFilePointerEx ← fflush`), while POSIX is unaffected.

This applies at minimum to the dry-run subprocess (`python -m robot --dryrun`) and the
library-availability `pip list` check; it does not apply to the full-run path, which
executes Robot Framework in-process (`run_cli`) and spawns no child.

#### Scenario: dry-run subprocess does not inherit stdin

- **WHEN** the server runs a suite in dry-run mode via the `python -m robot --dryrun` subprocess
- **THEN** that subprocess is invoked with a non-inheriting stdin, so on Windows it completes in about one second instead of deadlocking until the dry-run timeout, and Linux behaviour is unchanged

#### Scenario: library-check subprocess does not inherit stdin

- **WHEN** the server checks installed packages via a `pip list` subprocess
- **THEN** that subprocess is invoked with a non-inheriting stdin (the same inherited-stdin deadlock, with a shorter timeout, cannot occur)

### Requirement: Subprocess timeouts reap the whole process tree

When a serving-time subprocess exceeds its timeout, the server SHALL terminate the child
AND all of its descendants (Windows `taskkill /F /T /PID`; POSIX process group `killpg`),
because killing only the direct child can orphan a grandchild (e.g. a uv-trampoline launcher
whose real interpreter survives) that keeps the captured stdout/stderr pipes open and makes
the post-kill `communicate()` hang. The reaper MUST be best-effort and MUST NOT raise.

#### Scenario: a hung dry-run is reaped cleanly on timeout

- **WHEN** a dry-run subprocess does not exit before its timeout
- **THEN** the server kills the child and its descendants, returns a structured "timed out" error, and does not itself hang draining pipes held open by an orphaned grandchild

#### Scenario: reaper failure is non-fatal

- **WHEN** the process-tree kill mechanism is unavailable or fails (e.g. `taskkill` missing, permission denied)
- **THEN** the server falls back to killing the direct child and still returns the timeout error without raising
