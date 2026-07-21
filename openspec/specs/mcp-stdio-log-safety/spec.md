# mcp-stdio-log-safety Specification

## Purpose
TBD - created by archiving change mcp-stdio-log-safety. Update Purpose after archive.
## Requirements
### Requirement: fd 1 is never redirected while the MCP transport owns it

robotmcp SHALL NOT redirect file descriptor 1 (the JSON-RPC stdio channel) at the
OS level during keyword or context execution. Protection of fd 1 from Robot
Framework console output SHALL rely on RF `console='none'` and a Python-level
stdout redirect that does not alter fd 1.

#### Scenario: a response write during keyword execution reaches the client
- **WHEN** a JSON-RPC response is written to fd 1 while a keyword / RF context operation is executing
- **THEN** the bytes are written to the actual stdio channel (fd 1), not misrouted to stderr

#### Scenario: RF context lifecycle produces no fd-1 output
- **WHEN** `start_suite`, `start_test`, and `end_test` run with `console='none'` and without any fd-level stdout suppression
- **THEN** zero bytes are written to fd 1

### Requirement: stderr logging defaults to WARNING and is configurable

robotmcp SHALL log to stderr through a handler whose default level is WARNING,
overridable via the `ROBOTMCP_LOG_LEVEL` environment variable, with a formatter
that includes the record level.

#### Scenario: default run is quiet
- **WHEN** robotmcp runs with no `ROBOTMCP_LOG_LEVEL` set
- **THEN** INFO and DEBUG records are not emitted to stderr; WARNING and above are

#### Scenario: verbose troubleshooting is opt-in
- **WHEN** `ROBOTMCP_LOG_LEVEL=INFO` (or DEBUG) is set
- **THEN** INFO (or DEBUG) records are emitted to stderr

### Requirement: stderr writes never block execution

The stderr logging handler SHALL NOT allow a blocked or broken stderr pipe to
propagate an error into keyword execution; such records SHALL be dropped.

#### Scenario: a blocked stderr pipe drops the record
- **WHEN** writing a log record to stderr raises `BlockingIOError` or `BrokenPipeError`
- **THEN** the record is dropped and no exception propagates to the caller

### Requirement: startup readiness feedback

On startup, before serving, robotmcp SHALL emit one concise, always-visible line
to stderr indicating the server is ready and summarizing its key configuration
(version, transport, effective log level, whether MCP log notifications are on,
attach status, and available test libraries). This line SHALL NOT be written to
stdout and SHALL be visible regardless of `ROBOTMCP_LOG_LEVEL`.

#### Scenario: a ready line with config is emitted on stderr
- **WHEN** the server starts and is about to serve
- **THEN** a single stderr line reports readiness and the current configuration, and nothing is written to stdout for it

### Requirement: opt-in structured MCP log notifications

robotmcp SHALL provide an opt-in channel, enabled by
`ROBOTMCP_MCP_LOG_NOTIFICATIONS`, that forwards Python log records to the MCP
client as structured `notifications/message` with a mapped level, without blocking
the calling thread or the event loop. When disabled or when no client session is
available, it SHALL have no effect.

#### Scenario: disabled by default
- **WHEN** `ROBOTMCP_MCP_LOG_NOTIFICATIONS` is not set
- **THEN** no MCP log notifications are sent and logging behaves as stderr-only

#### Scenario: enabled forwards records at the correct level
- **WHEN** the flag is set, a client session is active, and a WARNING record is logged
- **THEN** a `send_log_message` is scheduled on the event loop with the MCP level mapped from WARNING, without the logging call blocking

#### Scenario: no session is a no-op
- **WHEN** the flag is set but no MCP session/loop is captured for the current work
- **THEN** the record is not forwarded and no error is raised

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

