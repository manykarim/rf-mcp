## ADDED Requirements

### Requirement: RF agent skills are bundled and installable across popular coding agents

rf-mcp SHALL make Robot Framework agent skills (from `rf-agentskills`, via an optional `rf-mcp[skills]`
extra) installable through its existing onboarding installer, writing skill folders in each supported
agent's native location (`.agents/skills/` for Codex/Cursor/opencode/Kilo/Goose/pi and `.claude/skills/`
for Claude Code). Skill installation SHALL be opt-in, SHALL NOT overwrite the user's existing skills/hooks
(merge, not clobber), and SHALL be cleanly uninstallable via a tracked manifest.

#### Scenario: one command provisions connection + skills for an agent
- **WHEN** a user runs the rf-mcp onboarding installer with skills enabled for a supported agent
- **THEN** the agent receives both the rf-mcp MCP connection config and the RF skill folders in that agent's native format, without clobbering existing configuration

#### Scenario: skills install is reversible
- **WHEN** the user uninstalls
- **THEN** only the files rf-mcp installed are removed (tracked by a hash manifest), leaving pre-existing user content intact

### Requirement: RF skills are served agent-agnostically over MCP

rf-mcp SHALL serve the bundled skills over the MCP connection as `skill://` resources (via the stable
fastmcp v3 skills provider), including per-skill integrity digests, and SHALL provide a tool-based
fallback (e.g. `list_skills`/`get_skill` or resources-as-tools) so agents that consume only MCP tools can
still retrieve skill content on demand. The server instructions SHALL signpost the available skill index.

#### Scenario: a resource-capable client reads skills over MCP
- **WHEN** an MCP client that consumes resources connects to rf-mcp
- **THEN** it can list and read the RF skills as `skill://` resources with integrity digests, with no separate file install

#### Scenario: a tools-only client can still get skills
- **WHEN** a client that consumes only MCP tools connects
- **THEN** it can discover the skill index and fetch a skill's content through a tool, and the WORKFLOW-GUIDE instructions point it there

### Requirement: Bundled skills are scoped to non-duplicative knowledge and validated for uplift

The bundled skill content SHALL be scoped to knowledge rf-mcp does not already deliver through its
instructions and guidance tools (avoiding redundant per-tool usage docs), and capability claims for the
bundle SHALL be backed by the control-vs-treatment evaluation harness rather than asserted.

#### Scenario: skills do not duplicate existing guidance
- **WHEN** a skill overlaps materially with rf-mcp's existing guidance (e.g. `get_locator_guidance`, tool docstrings)
- **THEN** it is excluded or reduced so the bundle adds workflow knowledge rather than repeating tool usage

#### Scenario: uplift is measured, not assumed
- **WHEN** the bundle is proposed as a capability improvement
- **THEN** the A/B eval harness has been run (control = rf-mcp only, treatment = rf-mcp + skills), and skills that show no measurable delta are pruned
