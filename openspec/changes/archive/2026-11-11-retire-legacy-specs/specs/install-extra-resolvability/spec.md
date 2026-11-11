## REMOVED Requirements

### Requirement: The `all` extra resolves with default resolver settings on every supported platform
**Reason**: The 0.x installer (`install`, `uninstall`, `list`, the tracked manifest, the attach bridge) and the `desktop` and `all` extras were removed in rf-mcp 2.0.

**Migration**: `rf-mcp init`, `rf-mcp sync-skills` and `rf-mcp doctor` are specified in `mcp-skills-provider`; the CI `installability` job resolves and installs the built wheel with its `otel` extra.

### Requirement: The `desktop` extra states its resolver and platform requirements
**Reason**: The 0.x installer (`install`, `uninstall`, `list`, the tracked manifest, the attach bridge) and the `desktop` and `all` extras were removed in rf-mcp 2.0.

**Migration**: `rf-mcp init`, `rf-mcp sync-skills` and `rf-mcp doctor` are specified in `mcp-skills-provider`; the CI `installability` job resolves and installs the built wheel with its `otel` extra.

### Requirement: Installability is verified the way a user installs, not as a local path
**Reason**: The 0.x installer (`install`, `uninstall`, `list`, the tracked manifest, the attach bridge) and the `desktop` and `all` extras were removed in rf-mcp 2.0.

**Migration**: `rf-mcp init`, `rf-mcp sync-skills` and `rf-mcp doctor` are specified in `mcp-skills-provider`; the CI `installability` job resolves and installs the built wheel with its `otel` extra.
