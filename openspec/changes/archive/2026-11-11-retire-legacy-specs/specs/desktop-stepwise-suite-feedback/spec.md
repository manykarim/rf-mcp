## REMOVED Requirements

### Requirement: Session step accounting
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: execute_step surfaces running counts
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: build_test_suite reports executed/recorded/failed statistics
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: Empty or near-empty suite warning
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.
