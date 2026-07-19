## ADDED Requirements

### Requirement: Expected step failures are signaled as a WARNING-level tool error

When an `execute_step` (or equivalent step-executing tool) fails for an expected
reason (a failed Robot Framework keyword, an unreachable attach bridge), rf-mcp
SHALL signal the failure as a FastMCP `ToolError` carrying a WARNING log level, so
the server logs a single WARNING line without a Python traceback, while genuinely
unexpected exceptions continue to be logged at ERROR with a traceback.

#### Scenario: a failed step logs one WARNING line, no traceback
- **WHEN** an execute_step call fails on a Robot Framework keyword and the FastMCP runtime is 3.x
- **THEN** the server logs the tool error at WARNING level with no Python traceback on stderr

#### Scenario: unexpected exceptions still surface a traceback
- **WHEN** an unexpected (non-tool) exception occurs during a tool call
- **THEN** it is logged at ERROR with a traceback (not downgraded to WARNING)

### Requirement: The failure payload is preserved for the agent

Signaling a failure as a tool error SHALL NOT reduce what the client/agent receives:
the result SHALL be marked as an error and SHALL carry the failure detail — the
Robot Framework error message, any suggested keyword / hint, and the step id.

#### Scenario: the agent still sees a failed step with actionable detail
- **WHEN** an execute_step call fails
- **THEN** the tool result is marked as an error (isError) and its content includes the RF error message, the suggested-keyword or a hint, and the step id

### Requirement: Version-safe error construction

rf-mcp SHALL construct the WARNING-level tool error through a single compat seam
that degrades safely if the running FastMCP does not support a per-error log level,
so a failed step never surfaces a construction error in place of its failure detail.

#### Scenario: fallback preserves the failure on an unsupported runtime
- **WHEN** the running FastMCP does not accept a per-error log level
- **THEN** the failure is still raised as a tool error carrying the original failure detail (not a TypeError about the log-level argument)

### Requirement: FastMCP 3.x dependency without changing the supported Python range

rf-mcp SHALL depend on FastMCP 3.x (required for the WARNING log-level mechanism)
while keeping its supported Python range and CI matrix unchanged unless the
resolved dependencies require otherwise.

#### Scenario: the dependency upgrade keeps Python support intact
- **WHEN** the project is pinned to fastmcp>=3.0 and the lock is regenerated
- **THEN** the supported Python range remains >=3.10 and the CI Python matrix is unchanged, because the FastMCP 3.x dependency chain also supports >=3.10
