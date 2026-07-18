## Context

Clean-room experiments (`experiments/uv-tool-install/`, Docker, no repo mounted)
established the current-state facts:

| Fact | Evidence |
| --- | --- |
| `uv tool install "<wheel>[web,api]"` works; extras-on-wheel accepted | 8s, 169 MB |
| `robotmcp` on PATH; stdio MCP `initialize`/`tools/list` (19 tools) | handshake probe |
| Requests / Selenium / Browser import in the tool venv | import probe |
| API works with zero post-install; Selenium works with a system browser; Browser works after `rfbrowser init` | restful-booker + saucedemo PASS |
| A coding agent wired as `{"command":"robotmcp"}` drove a web test to PASS | web-agent transcript |
| stdout stays pure JSON-RPC (0 contamination) | stderr probe |
| `rfbrowser` not on PATH, but `python -m Browser.entry init` works in-venv | verified |

Two reference installers were studied: **OpenSpec** (per-tool adapter registry
mapping each tool → skills path + commands path + scope, 31+ tools) and the
sibling **`rf-agentskills`** (`robotframework-agentskills`), which already ships a
cross-agent installer with agent detection, `--what skills,agents,hooks,mcp`,
`--scope project|user`, and a **hash-tracked manifest** enabling safe uninstall.
Per the maintainer, rf-mcp implements its **own standalone** installer (not reusing
rf-agentskills), but deliberately borrows that proven architecture.

Per-agent MCP config conventions (researched), the substrate for the adapter registry:

| Agent | Config file (project / user) | Format | Server entry |
| --- | --- | --- | --- |
| Claude Code | `.mcp.json` / `~/.claude.json` | JSON | `mcpServers.<n>{command,args,env}` |
| Codex | `.codex/config.toml` / `~/.codex/config.toml` | TOML | `[mcp_servers.<n>]` command/args/env |
| GitHub Copilot | `.vscode/mcp.json` | JSON | `servers.<n>{command,args,env}` |
| opencode | `opencode.json` / `~/.config/opencode/opencode.json` | JSON | `mcp.<n>{type:"local",command:[…],environment,enabled}` |
| Gemini CLI | `.gemini/settings.json` / `~/.gemini/settings.json` | JSON | `mcpServers.<n>{command,args,env,trust}` |
| Kilo Code | `.kilo/kilo.jsonc` / `~/.config/kilo/kilo.jsonc` | JSONC | `mcp.<n>{type,command,args,env}` |
| goose | — / `~/.config/goose/config.yaml` | YAML | `extensions.<n>{cmd,args,type:stdio,enabled,timeout}` |
| Cursor | `.cursor/mcp.json` / `~/.cursor/mcp.json` | JSON | `mcpServers.<n>{command,args,env}` |
| pi | **unconfirmed** — must be identified before shipping its adapter | ? | ? |

## Goals / Non-Goals

**Goals:**
- One command wires rf-mcp into whichever agents are on the machine, reversibly.
- Foundational tool-install UX (`init`/`doctor`/`--version`) so a manual path exists.
- An extensible `--what` + manifest so future skills/subagents/hooks slot in.

**Non-Goals:**
- No change to the MCP tool surface, server runtime, or session execution.
- Not bundling browsers into the wheel.
- Not authoring skills/subagents/hooks here — only reserving the installer plumbing.
- Not reusing rf-agentskills' codebase (maintainer chose standalone); alignment is
  by design pattern, not shared code.

## Decisions

1. **Adapter registry = one small class per agent.** Each adapter declares: agent id,
   detection probe(s), config path resolver for `project` and `user` scope, a
   format codec (JSON / JSONC / TOML / YAML), and a `render_server_entry(command,
   args, env)` that emits that agent's shape. Adding/repairing an agent is a
   localized change — important because these formats drift (Kilo migrated off
   `mcp_settings.json`; Codex has open config issues).

2. **Merge-in-place, never overwrite.** `install` reads the existing config, inserts
   the `robotmcp` server under the agent's key (JSON/JSONC/TOML/YAML round-tripped
   preserving comments/formatting where the codec allows), and writes back. If a
   `robotmcp` entry already exists it is updated only with `--force`, else reported.

3. **Hash-tracked manifest for safe uninstall.** A manifest (e.g.
   `~/.local/state/robotmcp/install-manifest.json`) records, per agent+scope, each
   file touched, whether rf-mcp created the whole file or only inserted a key, and a
   hash of the value it wrote. `uninstall` removes only entries whose current hash
   still matches; user-edited entries are left intact and reported. Mirrors the
   rf-agentskills manifest model. `--dry-run` prints the plan without writing.

4. **The registered command is the resolved `robotmcp` path.** Detection prefers the
   absolute path of the running executable (`shutil.which("robotmcp")` /
   `sys.argv[0]`) so the agent launches the tool-installed binary regardless of PATH
   context; `env` carries only opt-in vars (e.g. `ROBOTMCP_DISABLE_LEARNING`).

5. **Agent detection is best-effort and never gates.** Presence is inferred from the
   agent's config dir / binary. Interactive mode pre-checks detected agents; users can
   still target undetected ones explicitly. `--agents detected` installs only found
   ones; `--agents all` targets the full registry; `--agents <csv>` is explicit.

6. **`--what` is the extension seam.** v1 implements `mcp`. `skills`/`agents`/`hooks`
   are wired as recognized selectors that currently find no bundled assets and no-op
   with a note, so the CLI, manifest schema, and per-agent asset paths (reusing the
   OpenSpec/rf-agentskills path conventions) are ready when rf-mcp ships those.

7. **`init`/`doctor` invoke rfbrowser via `sys.executable -m Browser.entry init`**
   (verified), not PATH, so Playwright installs into the same env the tool-installed
   Browser library imports from. Advises (not fails) when the `web` extra or Node is
   absent.

8. **Keep `robotmcp`; add `rf-mcp` alias.** Renaming would break existing
   `{"command":"robotmcp"}` configs and the docker entrypoint; a second
   `[project.scripts]` entry is zero-risk.

## Risks / Trade-offs

- **Config-format drift across 9 agents** is the main maintenance cost. Mitigate with
  the isolated-adapter design + a per-adapter round-trip test fixture (sample config
  in → server inserted → parse-back asserts shape) so a format change fails loudly.
- **`pi` adapter is unconfirmed.** Ship the other 8 and mark `pi` as `status:
  planned` in the registry until its MCP-config convention is verified; `robotmcp
  list --agents` surfaces adapter status honestly.
- **Writing to user-global config files** (goose/Gemini/Codex/Kilo user scope) is
  higher-blast-radius. Default `--scope project` where the agent supports it; require
  explicit `--scope user` for global; always `--dry-run`-able; manifest makes it
  reversible.
- **JSONC/TOML/YAML round-tripping** can lose comments/formatting. Use
  comment-preserving codecs where practical (e.g. `tomlkit`, ruamel.yaml,
  json5/jsonc-aware) and document any lossy cases.
- **Harness caveat (not shipped):** the experiment's Claude Code 2.1.212 CLI
  intermittently refused `--dangerously-skip-permissions` as non-root on the fast API
  path; the web agent run and all deterministic MCP drives succeeded. A coding-agent
  CLI quirk, unrelated to rf-mcp.
