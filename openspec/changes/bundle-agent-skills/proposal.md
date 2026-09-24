> **DECISION (2026-08-12): STOPPED — not implementing §1–§3.**
> The §5 eval-gate (4 A/B evals, ~460 real sessions; `report.md`, `report_small.md`, `report_hard2x2.md`,
> `report_conf.md`) established that the RF agent skills add **≤0 value on top of rf-mcp** — rf-mcp and the
> skills are **substitutes, not complements** (the skill has standalone value only when rf-mcp is *absent*,
> and co-bundling is neutral-to-harmful). The premise below (co-install/serve skills alongside rf-mcp) is
> therefore not worth building. The one concrete defect found — the browser skill's wrong iframe combinator
> — is handed off to the rf-agentskills project in `docs/rf-agentskills-browser-skill-iframe-fix.md`.
> Kept for the record; not archived-as-done.

## Why

rf-mcp gives agents a Robot Framework *execution surface* (≈60 tools, 3 prompts, the WORKFLOW-GUIDE
instructions, and guidance tools like `get_locator_guidance`). What it deliberately does **not** ship is
the deep **knowledge/skills tier**: per-library workflow expertise (locator strategy, assertion engine,
iframes/shadow-DOM, auth storage, troubleshooting), suite/resource architecture, and output.xml
debugging. That tier already exists — `robotframework-agentskills` (same author, Apache-2.0): 12
Anthropic-spec `SKILL.md` skills (6 curated knowledge packs — the Browser skill alone is ~3,700 lines —
plus 6 script-based generators/analyzers), 4 subagents, and 4 hooks, with **zero code coupling to rf-mcp**
(so bundling is additive). Its own FAQ notes several skills "rely on rf-mcp for their primary value
proposition: letting the agent run the test it just wrote."

Two things make bundling worthwhile *now*: (1) file-based Agent Skills (agentskills.io open standard) is
the **only skill channel with ~10/10 coverage** across Claude Code, Codex, Cursor, opencode, Kilo, Goose,
and pi; and (2) the skills-over-MCP machinery rf-mcp needs is **already stable in `fastmcp>=3.0`**
(`SkillsDirectoryProvider` serves `skill://` resources; `ResourcesAsTools` bridges to tools-only clients) —
the shape the emerging MCP standard (SEP-2640 "Skills Extension") blesses. rf-mcp already *is*
skills-over-MCP in substance (instructions-as-pointer + guidance tools) but exposes **0 MCP resources** —
the one gap.

The goal: make `robotmcp init` provision, in one command, both the MCP connection (already shipped) **and**
the RF skills, across all popular agents — plus serve the same skills agent-agnostically over MCP.

## What Changes

- **Optional `rf-mcp[skills]` extra** depending on the published `rf-agentskills` package (single source of
  truth; no content duplication). rf-mcp consumes its canonical `SKILL.md` folders.
- **Installer co-installs skills** — extend `src/robotmcp/onboarding/adapters.py` so the existing per-agent
  installer also writes skill folders in each agent's native shape: `.agents/skills/` (Codex, Cursor,
  opencode, Kilo, Goose, pi) + `.claude/skills/` (Claude Code), reusing rf-agentskills' proven transforms,
  hash-manifest uninstall, and non-clobbering merges. Opt-in, scope-selectable.
- **Serve skills over MCP** — register `SkillsDirectoryProvider` (stable fastmcp v3) to expose the same
  folders as `skill://…` resources (+ sha256 `_manifest`), and add a tools bridge
  (`ResourcesAsTools` or a small `list_skills`/`get_skill` pair) for tools-only clients. Signpost the skill
  index from the WORKFLOW-GUIDE instructions.
- **Scope + prove** — curate skill exposure to knowledge rf-mcp does **not** already deliver (dedup vs
  `get_locator_guidance`/docstrings), and gate capability claims on the existing A/B eval harness
  (`rf-skill-eval`, control-vs-treatment, Haiku, Cliff's delta) — prune skills that show no delta.

Explicitly out of scope: MCP **prompts** as a skills channel (human-invoked, ~5/10 coverage); any
dependency on **fastmcp 4 beta** (everything ships on stable v3); waiting for SEP-2640 to merge before
shipping (files + v3 resources + tool bridge already give full coverage).

## Capabilities

### New Capabilities

- `agent-skills`: bundling and provisioning of Robot Framework agent skills with rf-mcp — installed into
  each agent's native format and served agent-agnostically over MCP, scoped to non-duplicative knowledge
  and validated for uplift.

## Impact

- `pyproject.toml` (`skills` extra → `rf-agentskills`), `src/robotmcp/onboarding/adapters.py` +
  `manifest.py` (co-install skills per agent), `src/robotmcp/server.py` (`SkillsDirectoryProvider` +
  tools bridge + instructions signpost). No change to existing MCP tools or the connection installer's
  current behavior (skills are additive/opt-in). Depends on the `rf-agentskills` PyPI package and the
  stable `fastmcp>=3.0` skills API already in use.
