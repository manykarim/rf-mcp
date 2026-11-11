# Spec Delta

## Purpose

The `tests/e2e/agent_eval` harness runs coding-agent CLIs on seeded Robot Framework projects and grades both
the outcome (suite passes, injected bugs caught) and the process (whether rf-mcp was used the intended way).

## ADDED Requirements

### Requirement: Codex CLI models run through codex exec
A model named `cx-<model>[@<N>k][~<effort>]` SHALL run through `codex exec --json` in the seeded project, with
reasoning effort `low` by default. `@<N>k` SHALL set the context window to N thousand tokens, with auto-compaction
at 80% of it. `cx-oss-<model>` SHALL run a local model through Codex's Ollama provider.

#### Scenario: Small-window model name
- **WHEN** the harness is run with `--models cx-gpt-5.6-luna@32k`
- **THEN** the command is `codex exec` with model `gpt-5.6-luna`, a 32,000-token context window, auto-compaction at 25,600 tokens and effort `low`

#### Scenario: Local model name
- **WHEN** the harness is run with `--models cx-oss-qwen3_8b`
- **THEN** the command uses Codex's local Ollama provider with model `qwen3:8b`

#### Scenario: Local model metadata
- **WHEN** a `cx-oss-*` model runs and Codex has a cached model catalog
- **THEN** Codex gets a catalog entry for that model with direct (not code-mode) tools, skills usage instructions and the configured window

### Requirement: OpenRouter models run through Claude Code and Codex
A model named `or-<vendor>.<model>` SHALL run Claude Code on OpenRouter with every Claude Code model role set to
`<vendor>/<model>`, and `cx-or-<vendor>.<model>` SHALL run Codex on OpenRouter's Responses API with a model
catalog entry that keeps `apply_patch`. The OpenRouter key SHALL be read from the environment or the repo's
git-ignored `.env`, SHALL be passed only to the agent process, and SHALL NOT appear in the printed commands.

#### Scenario: Claude Code on OpenRouter
- **WHEN** the harness runs `or-z-ai.glm-5.3-flash`
- **THEN** `claude -p --model z-ai/glm-5.3-flash` runs with OpenRouter's Anthropic-compatible base URL and no request uses an Anthropic model

#### Scenario: Key stays with the agent
- **WHEN** an OpenRouter model runs in the `mcp` condition
- **THEN** the MCP server and grader environments do not contain `OPENROUTER_API_KEY`

### Requirement: Codex runs are unattended and isolated from user configuration
Codex runs SHALL use the `workspace-write` sandbox with network access and `approval_policy="never"`, SHALL
pre-approve the rf-mcp tools in the project's `.codex/config.toml`, SHALL disable the user's apps, plugins
and other features that add tools or context unrelated to the scenario, and SHALL give the agent a shell the
sandbox can start.

#### Scenario: Shell on Windows
- **WHEN** a Codex agent runs a shell command on Windows where the only PowerShell 7 is the Store app
- **THEN** the command runs in System32 `powershell.exe` instead of failing with access denied

#### Scenario: rf-mcp tool under approval policy never
- **WHEN** a Codex agent calls `rf_run` in the `mcp` condition
- **THEN** the call runs instead of being rejected for missing approval

### Requirement: Skill sync setup
The `mcp+sync` setup SHALL run `rf-mcp sync-skills` for the agent into the project before the run, and results
SHALL be labeled `mcp-sync`.

#### Scenario: Codex with synced skills
- **WHEN** a Codex run uses `--conditions mcp+sync`
- **THEN** the project has the rf-mcp skills under `.agents/skills` and the result name ends in `__mcp-sync__r<N>`

### Requirement: Codex transcripts are graded like Claude transcripts
The harness SHALL map Codex `--json` events onto the call model the graders use: MCP calls as
`mcp__<server>__<tool>`, shell commands as Bash or PowerShell with the wrapper removed, file changes as
Write or Edit, and failed calls or non-zero exit codes as errors. Input tokens per run SHALL appear in the report.

#### Scenario: Codex PowerShell wrapper
- **WHEN** a Codex transcript contains a command execution of `"C:\Program Files\PowerShell\7\pwsh.exe" -Command "robot tests"` that exits with 1
- **THEN** the graders see a failed PowerShell call with the command `robot tests`

### Requirement: Unverified finishes are flagged
The process graders SHALL flag `unverified` when the agent wrote a `.robot` or `.resource` file (write tools,
`rf_repl save`, `files` on a run or session, or a shell redirect) and no test run passed after the last such
write. A dry run SHALL NOT count as passed, and `files` on `rf_analyze` SHALL NOT count as a write.

#### Scenario: Suite written, only analyzed
- **WHEN** a transcript writes `tests/a.robot` and then only calls `rf_analyze`
- **THEN** the process flags include `unverified`
