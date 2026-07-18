## Why

Users want to install rf-mcp the way modern Python CLIs are distributed —
`uv tool install rf-mcp` — and then wire it into whichever coding agent they use,
without hand-editing each agent's bespoke MCP-config file. Clean-room experiments
(`experiments/uv-tool-install/`) proved the packaging is **already sound**: `uv
tool install "rf-mcp[web,api]"` yields a working `robotmcp` command that speaks MCP
over stdio (19 tools), all RF test libraries import, the stdio channel stays clean
(0 stdout contamination), and a real coding agent (MiniMax-M3, wired as
`{"command":"robotmcp"}`, no repo) drove a headless-browser web test to a PASS. The
API path works end-to-end against restful-booker with **zero** post-install steps.

Two barriers remain, both about ergonomics rather than capability:

1. **First-run friction.** Bare `uv tool install rf-mcp` installs core only (no test
   libraries), the extras (`[api]`/`[web]`/`[all]`) are undiscoverable, and the
   Browser/Playwright path needs `rfbrowser init` whose executable is not on PATH.
2. **Per-agent wiring is manual and inconsistent.** Every agent stores MCP servers
   in a different place and format — Claude Code `.mcp.json`, Codex
   `~/.codex/config.toml` `[mcp_servers]`, opencode `opencode.json` `mcp`, Gemini
   `~/.gemini/settings.json` `mcpServers`, Kilo `~/.config/kilo/kilo.jsonc` `mcp`,
   goose `~/.config/goose/config.yaml` `extensions`, Cursor `.cursor/mcp.json`, etc.
   Users must know all of these. Formats also drift (Kilo migrated off
   `mcp_settings.json`; Codex has open config bugs), so ad-hoc docs rot.

This change makes rf-mcp self-installing into coding agents: `uv tool install
"rf-mcp[all]"` → `robotmcp install` detects the agents on the machine and writes the
correct MCP registration for each, with a safe, reversible `uninstall`. The design
mirrors the proven pattern in the sibling project `robotframework-agentskills`
(`rf-agentskills`): agent detection, per-agent adapters, and a hash-tracked manifest
so uninstall removes only what it wrote and never clobbers user edits.

## What Changes

**Foundation — rf-mcp as a first-class tool**
- `robotmcp init [--browsers]`: detect installed test libraries; run the bundled
  `rfbrowser init` **in the tool's own venv** (`sys.executable -m Browser.entry
  init`); check Node.js; print the ready MCP config. Idempotent, non-destructive.
- `robotmcp --version` and read-only `robotmcp doctor` (version + per-library import
  status + browser-init state + Node presence).
- Add an `rf-mcp` console-script alias alongside `robotmcp` (both work).

**Multi-agent installer (standalone in rf-mcp)**
- `robotmcp install` / `uninstall` / `list`, with an **agent-adapter registry** for:
  Claude Code, Codex, GitHub Copilot, opencode, Gemini CLI, Kilo Code, pi, goose,
  Cursor. Each adapter knows that agent's config path(s), file format (JSON / JSONC
  / TOML / YAML), the server-entry shape, and project-vs-user scope.
- **Agent detection** (which agents are present) with an interactive pre-checked
  selection, plus scriptable flags: `--agents all|detected|<csv>`,
  `--scope project|user`, `--dry-run`, `--yes/--no-input`, `--force`.
- **`--what mcp,skills,agents,hooks`** selects what to install. v1 implements `mcp`
  (register the rf-mcp server); `skills`/`agents`/`hooks` are declared, no-op stubs
  now so the CLI and manifest are ready when rf-mcp starts shipping those assets.
- A **hash-tracked manifest** records every file/edit written per agent; `uninstall`
  reverts only entries whose current hash still matches (edited-by-user entries are
  left and reported), so it is safe and idempotent.

**Docs & release hygiene**
- README "Install into your coding agent" section: the extras matrix, `robotmcp
  install`, the supported-agent table, and the manual `{"command":"robotmcp"}`
  snippet as a fallback.
- Bump the version and publish current work to PyPI (the published release is stale).

Non-goals: no change to the MCP tool surface or session runtime; not bundling
browsers into the wheel; not authoring skills/subagents/hooks in this change (only
reserving the installer plumbing for them).

## Capabilities

### New Capabilities
- `tool-install-onboarding`: how rf-mcp is installed as a standalone CLI tool and
  registered into coding agents — the `init`/`doctor`/`--version` surface, the extras
  and browser-init story, and the multi-agent `install`/`uninstall`/`list` installer
  (adapter registry, detection, scope, `--what` selection, hash-tracked manifest).

### Modified Capabilities
<!-- none -->
