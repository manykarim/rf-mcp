# Spec Delta

## ADDED Requirements

### Requirement: Codex wiring explains skill sync and tool approval
The configuration that `rf-mcp init --agent codex` writes or prints SHALL tell the user to run
`rf-mcp sync-skills --agent codex`, because Codex shows the model neither MCP server instructions nor tool
descriptions. It SHALL include `default_tools_approval_mode = "approve"` only as a commented, opt-in line that
says rf-mcp tools then run outside Codex's sandbox.

#### Scenario: Generated Codex config
- **WHEN** a user runs `rf-mcp init --agent codex`
- **THEN** the `[mcp_servers.rf-mcp]` entry is followed by comments naming `rf-mcp sync-skills --agent codex` and a commented `default_tools_approval_mode = "approve"` line, and the approval setting is not active
