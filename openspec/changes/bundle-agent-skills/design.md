## Context

rf-mcp ships an execution surface (tools/prompts/instructions/guidance tools) but not the deep RF
knowledge tier. `robotframework-agentskills` (same author, Apache-2.0, published as `rf-agentskills` on
PyPI) provides 12 `SKILL.md` skills + subagents + hooks, already distributed to 7 agents by its own
installer, with an unrun A/B eval harness. rf-mcp's onboarding installer already writes MCP config per
agent; `fastmcp>=3.0` (already a dependency) already ships `SkillsDirectoryProvider` + `ResourcesAsTools`.
The emerging standard is SEP-2640 (`skill://` resources, In Review). rf-mcp exposes 0 MCP resources.

## Goals / Non-Goals

**Goals:** one-command provisioning of connection + skills across all popular agents (files-first for
universal coverage); agent-agnostic skills-over-MCP on stable v3; non-duplicative, eval-validated skill
scope; forward-compatible with SEP-2640.

**Non-Goals:** MCP prompts as a skills channel; any fastmcp-4 beta dependency; blocking on SEP-2640
merge; reimplementing rf-agentskills' per-agent transforms (reuse them); shipping subagents/hooks as a
requirement (skills are the core; subagents/hooks are optional follow-on).

## Decisions

**D1 — Depend, don't vendor (chosen).** Add an optional `rf-mcp[skills]` extra → `rf-agentskills`. Single
source of truth; rf-mcp consumes its canonical `SKILL.md` folders (locate via the installed package's
resources path). Pin a compatible range and align release notes. Trade-off: coupling to an early-maturity
package + version alignment — accepted for zero content duplication.

**D2 — Files-first, dual-channel (the coverage reality).** Files are the only 10/10 channel:
- Installer writes `.agents/skills/` (Codex, Cursor, opencode, Kilo, Goose, pi) + `.claude/skills/`
  (Claude Code), reusing rf-agentskills' transforms/hash-manifest/non-clobber merge (absorb via the
  package or a thin adapter). This is the primary, guaranteed-coverage path.
- Server registers `SkillsDirectoryProvider` → `skill://` resources (+ sha256 `_manifest`) for
  resource-capable clients (Claude Code today; SEP-2640-shaped for the future).
- A tools bridge (`ResourcesAsTools`, or a minimal `list_skills`/`get_skill`) covers tools-only clients
  (Codex, opencode). The WORKFLOW-GUIDE instructions signpost the skill index (SEP-2640 endorses
  instructions-as-signpost).

**D3 — Stable v3 only.** All skills machinery is in `fastmcp>=3.0`; do not add a fastmcp-4 dependency
(beta, no GA). Later v4 migration is modest and out of scope here.

**D4 — Scope + prove (dedup + eval-gate).** Curate exposure to knowledge rf-mcp doesn't already serve
(dedup vs `get_locator_guidance`/docstrings — LlamaIndex: overlapping skills are "rarely invoked, no
better result"). Run `rf-skill-eval` (control vs treatment, Haiku to avoid delta compression, Cliff's
delta, leave-one-out) and prune skills with no delta before claiming uplift.

## Risks / Trade-offs

- **Unmeasured uplift** — eval harness exists but unrun; the eval-gate (D4) is the mitigation. Do not
  market a capability gain before numbers.
- **Redundancy/token noise** — mitigated by D4 dedup; skills are progressive-disclosure (metadata-only
  until activated), so cost is low if scoped.
- **SEP-2640 churn** — `skill://` URI/method/digest semantics may change; build to the draft, keep the
  resource layer thin.
- **Per-agent path drift** — conventions still settling (Goose reorg; Claude Code won't read
  `.agents/skills/`; pi rejects MCP → files only). Pin exact paths at install; keep the adapter matrix
  updatable.
- **Packaging coupling** — `rf-agentskills` is early-maturity; version-align and pin. Staleness between
  installed files and rf-mcp version → use the `_manifest` digests for a session-start staleness check.
- **Windows install fiddliness** — rf-agentskills needed bash→Node hook fixes; validate skill install +
  hook merge on Windows.

## Migration Plan

Additive and opt-in: no change to existing tool/connection behavior. `rf-mcp[skills]` is off by default;
users opt in at install. Phased: (1) build (extra + installer + provider + bridge + signpost),
(2) validate (run the eval, prune), (3) track the standard (declare the `io.modelcontextprotocol/skills`
extension when SEP-2640 merges; migrate to fastmcp 4 at GA).

## Open Questions

- Absorb rf-agentskills' adapter transforms into rf-mcp vs call the `rf-agentskills` installer as a
  library — prefer calling it if it exposes a stable API, to avoid a second copy of the matrix.
- Whether to also bundle the 4 subagents/hooks or ship skills-only in v1 (lean skills-only; revisit after
  eval).
