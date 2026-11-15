# Proposal

## Why

rf-mcp reads project skills (`.robotframework/skills/`) once at startup, from the folder the server starts in. VS Code
1.145.2 starts a user-level `rf-mcp` entry in the user's home folder (measured on 2026-11-12, task 6.4 of
`project-independent-server`), so a global install registered once for all projects never serves a project's
skills there. Tool calls already find the project through the client's workspace folders (MCP roots); skills should
find it the same way.

## What Changes

- Project skills are looked up per request, for the same project root a tool call would use: the pinned `--root`,
  else the client's workspace folders, else the startup folder walked up to the nearest project marker.
- With several workspace folders, `list_resources` serves the project skills of every folder that has them. The
  first folder wins when two define a skill of the same name. Tool calls still stop and ask for `root`, as now.
- External (`--skills-dir`) and built-in skills are unchanged and served for every project.
- The server instructions are sent before the client can report its workspace folders. When rf-mcp starts outside
  any project (no marker found and no pinned root), the instructions say that project skills appear in
  `list_resources` instead of listing them.
- The spec's skill-root precedence is corrected to what the code does: project skills first, then `--skills-dir`
  folders, then built-in skills; the first skill of a name wins.
- Docs: the VS Code note in `docs/CONFIGURATION.md` (project skills not served for a user entry) is replaced by the
  new behaviour.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `mcp-skills-provider`: "Layered skill roots for ecosystem plugins and project customizations" changes to resolve
  the project skill root per request from the call's project root, with the corrected precedence; a new
  requirement covers the instructions note for a server started outside any project.

## Impact

- Code: `src/robotmcp/skills/provider.py` (a provider that picks the project skill folder per request and delegates
  to one `SkillsDirectoryProvider` per folder), `src/robotmcp/client_roots.py` (shares its per-session roots cache
  with the skills provider), `src/robotmcp/server.py` (wiring).
- No new dependencies. No change for agents that start rf-mcp in the project, or for a pinned `--root` (Docker,
  remote runners).
- Not in scope: the official Robot Framework skills package announced for
  [MarketSquare/robotframework-plugins](https://github.com/MarketSquare/robotframework-plugins). It is an external
  skill source (`--skills-dir`) and changes which collections the docs recommend, not where project skills come
  from; it belongs with the review of `bundle-agent-skills`.
