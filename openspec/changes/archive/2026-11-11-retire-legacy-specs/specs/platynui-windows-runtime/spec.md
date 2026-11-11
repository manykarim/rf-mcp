## REMOVED Requirements

### Requirement: Desktop automation never leaves keyboard modifiers held
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: Desktop keyword queries fail fast and honor timeout_ms
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: Desktop keyword execution never blocks the server event loop
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: The desktop safety guard is platform-aware on Windows
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: Held keys are recovered after a hard process termination
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.
