## REMOVED Requirements

### Requirement: Desktop suites carry a replay-environment preamble
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: User-defined suite setup takes precedence
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.

### Requirement: Preamble executes before the PlatynUI runtime exists
**Reason**: rf-mcp 2.0 no longer executes keywords in its own process and no longer ships the PlatynUI desktop tooling, stepwise suite recording or the batch executor this requirement governed.

**Migration**: Desktop tests run through `rf_run`, `rf_repl` and `rf_debug` like any other suite (see `robotcode-mcp-tools`), for example in the VNC image (`docs/DOCKER.md`). rf-mcp 0.36 keeps the desktop tooling at the `v0.36.0` tag.
