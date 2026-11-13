# Design

## Context

- `mount_skills_provider` (`src/robotmcp/skills/provider.py`) calls `resolve_skill_roots(config)` once, at startup.
  That puts `<startup root>/.robotframework/skills` (expanded), then the `--skills-dir` folders, then the built-in
  folder into one `SkillsDirectoryProvider(roots=..., reload=True)`. `SkillsDirectoryProvider` keeps the first skill
  of each name. `reload=True` rescans the folders on every request, but the folder list itself is fixed.
- `ClientRoots` (`src/robotmcp/client_roots.py`) is the middleware that fills `root` for `rf_*` tool calls. It asks
  handshake-era clients for their roots once per session, caches the answer, and clears the cache on
  `notifications/roots/list_changed`. `choose_root` picks one root and raises `ToolError` when several are
  ambiguous.
- FastMCP 4.1.8 providers answer `_list_resources`, `_get_resource`, `_list_resource_templates` and
  `_get_resource_template`. Inside a request, `fastmcp.server.dependencies.get_context()` returns the request's
  `Context`, including during `list_resources` / `read_resource` calls made through the `ResourcesAsTools` bridge.
- The `instructions` go out in the `initialize` result, before the client may receive server requests, so they
  can never depend on the client's roots.

## Goals / Non-Goals

**Goals:**

- Skills resolve their project root by the same rules and the same roots answer as tool calls, from one cache.
- Concurrent sessions with different workspace folders each see their own project skills.

**Non-Goals:**

- Sending `notifications/resources/list_changed` when the workspace folders change. Clients list resources again
  when they need them; the next listing is already correct.
- Recommending or bundling external skill collections (for example the coming MarketSquare package); see the
  proposal.
- Changing how `rf-mcp sync-skills` or `doctor` find skills. Both run in a shell in the project.

## Decisions

### D1. One `SkillsDirectoryProvider` per folder list, chosen per request

A new provider (`ProjectSkillsProvider`) implements the four resource hooks. Each call works out the folder
list for the current request (D2) and delegates to a `SkillsDirectoryProvider(roots=<list>, reload=True)` cached
under that list (a tuple of resolved paths).

- Keeping the whole list in one `SkillsDirectoryProvider` reuses its first-name-wins rule, so precedence across
  project, external and built-in skills stays exactly as today.
- Alternative: one provider whose `_roots` is swapped per request. Rejected: `_discover_skills` clears and refills
  shared state, so two sessions listing at the same time would see each other's skills.
- Alternative: stack separate providers in an `AggregateProvider`. Rejected: duplicates across child providers
  are resolved by version, not by order, so the project skill would not reliably win.
- The cache holds one entry per distinct folder list. Each session normally has one list, so the cache stays
  small; it is not bounded.

### D2. The folder list follows the tool-call rules

`resolve_skill_roots` gains a `project_roots` argument: the list of project roots for the request.

- A pinned `workspace_root` gives `[pinned]`.
- In server mode with no pinned root, the client's roots come from `ClientRoots` through a new public
  `await client_roots.roots_for(ctx)`, the same cached lookup `on_call_tool` uses.
- With no roots (no request context, a modern-protocol client, a client without roots, a timeout or an error),
  the list is `[startup_root()]`.
- With several roots, all of them are used in the client's order. A failing resource listing would hide every
  skill, so ambiguity is not an error here. Tool calls keep raising, because they need exactly one root.
- Client mode never asks for roots, as for tool calls.

### D3. A note in the instructions for a server started outside any project

`startup_root()` already reports its source. With no pinned root and source `cwd` (no marker found), the
instructions add one line: "Project skills: list_resources shows them once your workspace is known." The catalog
still lists the built-in and external skills. This costs about 80 characters of the 2,500-character budget. The
existing truncation keeps the total under the limit.

- Alternative: always add the line. Rejected: in a project the catalog already lists the project's skills, and
  every character of the instructions is read on every session.

### D4. Skill read metrics and the tool bridge are unchanged

`SkillReadMetrics` matches `skill://` URIs and `ResourcesAsTools` calls the server's resource listing, so both work
through the new provider without changes.

## Risks / Trade-offs

- [A client that advertises roots but never answers `roots/list`] → The existing 5 s timeout in `ClientRoots` also
  applies to the first skill listing; the result is cached, so only the first request of a session waits.
- [The first `list_resources` of a session costs one `roots/list` round trip] → It is the same lookup the first
  tool call needs, and it is shared through one cache.
- [A workspace folder on another machine (remote server, local client)] → The folder does not exist on the server,
  `SkillsDirectoryProvider` skips missing folders, and external and built-in skills are still served.
- [Unbounded provider cache] → Distinct folder lists come only from the clients' own workspace folders. If this
  ever grows, key the cache by session instead.

## Migration Plan

No migration. Agents that start rf-mcp in the project and pinned deployments see the same skills as before.
