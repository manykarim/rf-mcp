# Design

## Context

- `src/robotmcp/skills/official.py` has one loader, `load_official_skills()`, cached for the process. It imports `robotframework_plugins.bundle`, checks `CATALOG_FORMAT` against `SUPPORTED_FORMATS = {1}`, and reads `bundle.skills_dirs()` and `bundle.catalog()`. Its three callers (`resolve_skill_roots`, `build_server_instructions`, `official_entries` via `catalog_entries`) all have the `ServerConfig` at hand.
- The package (2026.11.0.dev1) also exports `fetch(ref="stable", *, cache_dir=None, timeout=30.0) -> Snapshot`, plus `Snapshot(ref, commit, path)` with `catalog()` and `skills_dirs()`, and `FetchError`.
  - A full SHA is taken as is; a branch or tag is resolved through `api.github.com`.
  - The commit's tarball comes from `codeload.github.com`, every entry is checked, and only `plugins/` is extracted.
  - The catalogue is built with the installed wheel's rules, and the result moves into `<cache>/<sha>/` in one step.
  - The cache is `%LOCALAPPDATA%\robotframework-plugins\snapshots` or `$XDG_CACHE_HOME`/`~/.cache/robotframework-plugins/snapshots`.
  - A snapshot of `main` measured 1.1 MB and 171 files, and was fetched in 1.4 s on 2026-11-16.
- A snapshot used as `--skills-dir` already works today (checked against `main`, commit `56547f0`): rf-mcp reads its `plugins/*/catalog.yaml` and serves all nine skills.
- The server is built by `create_server(config)` and run from `robotmcp.entry.main`. Tests build servers with `create_server` in memory and must never touch the network.

## Goals / Non-Goals

**Goals:**
- A ref channel never makes startup slower or dependent on the network.
- Only the package's public `bundle` API touches GitHub; rf-mcp adds no downloader of its own.
- Every failure ends in a working server: the stored snapshot, else the wheel's content.

**Non-Goals:**
- Swapping skills during a running session (see D5).
- Mirrors, tokens and proxy settings for the download. These belong to the package (MarketSquare/robotframework-plugins#22).
- Reading the agents' own marketplace clones (`~/.claude/plugins/marketplaces/...`).
- Pruning old snapshots from the package's cache.
- A configurable cache folder.

## Decisions

### D1. Snapshots come from the package's `fetch()`
rf-mcp calls `bundle.fetch(ref)` and serves the returned `Snapshot` like the bundled content.

Alternatives considered:
- **Running `git clone`/`git pull`**: needs git on every machine, and leaves rf-mcp responsible for a working tree.
- **rf-mcp's own HTTP download**: duplicates the package's archive checks and catalogue rules.
- **Reading the agents' marketplace clones**: covers only agents that have a marketplace, each with its own folder layout and update behaviour. If the plugins are also installed, the agent already loads the skills natively, so rf-mcp would serve them twice.

The package's fetch keeps one definition of "a valid snapshot" for the installer and rf-mcp, and improvements such as #22 reach rf-mcp with a version bump.

### D2. rf-mcp records which commit it fetched for each ref
A state file `<cache base>/rf-mcp/official-skills.json` maps each ref to its last fetch:
```json
{"refs": {"latest": {"commit": "56547f0e…", "path": "C:\\…\\snapshots\\56547f0e…",
                     "fetched_at": "2026-11-16T12:00:00Z", "checked_at": "2026-11-16T12:00:00Z",
                     "catalog_format": 1, "package_version": "2026.11.0.dev1"}}}
```
- The cache base is the same platform folder the package uses (`%LOCALAPPDATA%`, else `$XDG_CACHE_HOME` or `~/.cache`).
- At startup rf-mcp builds `Snapshot(ref, commit, path)` from this entry, after checking that `path/catalog.json` exists. Both the dataclass and its methods are public.
- Writes go to a temporary file in the same folder, followed by `os.replace`.

Alternatives considered:
- **Calling `fetch(commit)` at startup**: network-free on a cache hit today, but that's an implementation detail, and a miss would download during startup.
- **Reading the package's `<sha>/snapshot.json` to find the newest snapshot of a ref**: that file isn't part of the public API.

### D3. The loader takes the channel
`load_official_skills(channel)` is cached per channel, and callers pass `config.official_skills`.
- For `bundled`, it behaves exactly as today.
- For a ref, it uses the snapshot from D2 when the recorded `catalog_format` is in `SUPPORTED_FORMATS` and equals the running package's `CATALOG_FORMAT`. Otherwise it falls back to the bundled content and sets a `warning`, which the log and `doctor` show.

`OfficialSkills` gains `channel`, `commit`, `fetched_at`, `path` and `fallback_reason`. The format check uses rf-mcp's own record until the package records the format in snapshots (MarketSquare/robotframework-plugins#23). Then it reads `Snapshot.catalog_format` instead.

### D4. A background refresh, started only by the real server
`robotmcp.entry.main` starts the refresh after `create_server`, so in-memory test servers never fetch. It's a daemon thread with one attempt per process:
1. Skip if the channel is `bundled` or a full SHA, if `RF_MCP_OFFICIAL_SKILLS_REFRESH` is `0`, or if the ref's `checked_at` is newer than the interval.
2. Write `checked_at = now` first. When several agents start rf-mcp at once, the later ones then see a fresh check and skip, which keeps the GitHub API within its 60 unauthenticated calls per hour.
3. Call `bundle.fetch(ref, timeout=30)`. On success, record the commit, path, `fetched_at`, catalogue format and package version. On `FetchError` or any other exception, log one warning and keep the old entry. The `checked_at` stays, so an offline machine retries after one interval, not on every start.

`rf-mcp update-skills` runs the same fetch-and-record step in the foreground and ignores the interval.

### D5. New content applies at the next start
The server instructions (the official skills line and the command mapping) and the skill catalogue that `project_skills` matches against are built for the process. Serving a different set halfway through a session would contradict what the agent was told at the start. The refresh therefore only updates the state file.

A long-running HTTP server keeps its content until it restarts. The docs say so, and `update-skills` plus a restart is the explicit path. This is also how Claude Code applies marketplace updates and Nuclei applies template updates.

### D6. Channel values
- `bundled` is the default.
- A 40-hex-character value is a commit: it's never refreshed, and `update-skills` fetches it once.
- Anything else is a branch or tag, passed to `fetch()` as is.

rf-mcp doesn't validate refs itself; the package reports an unknown ref as a `FetchError`. `stable` is the recommended channel in the docs once it exists (MarketSquare/robotframework-plugins#21). `main` and `latest` are documented as unreleased content.

### D7. CLI
- **`rf-mcp update-skills [--ref REF]`** is a new subcommand next to `sync-skills`. Without `--ref`, it uses the configured channel from `RF_MCP_OFFICIAL_SKILLS`. It prints JSON (`ref`, `commit`, `path`, `skills`), and exits 1 on a failed fetch or when no ref is given and the channel is `bundled`.
- **`rf-mcp init --official-skills CHANNEL`** adds `RF_MCP_OFFICIAL_SKILLS` to the environment that `agent_config.py` already writes (`DEFAULT_ENV`), for every agent and both scopes.
- **`rf-mcp doctor`** reads the state file without touching the network. It reports the channel and what's served, as the spec describes, in both the text and the `--json` output (`official_skills.channel`, `commit`, `fetched_at`, `fallback_reason`).

### D8. Tests stay offline
- `tests/conftest.py`'s fake bundle gains `fetch`, `Snapshot` and `FetchError`. The fake `fetch` builds a snapshot folder from the same plugin specs, or raises.
- The state file goes to a temporary cache base, set by monkeypatching the base-folder function.
- One live test runs only with `RF_MCP_LIVE_FETCH=1` against the real package and GitHub, like `test_official_package.py` runs only when the package is installed.

## Risks / Trade-offs

- [GitHub's 60 unauthenticated API calls per hour are shared behind a company NAT] → at most one check per ref per interval per machine, `checked_at` written before fetching, and a failure is harmless. Mirrors and tokens come with #22.
- [A company proxy re-signs TLS and Python doesn't trust its CA] → the fetch fails and the wheel's content is served. The docs name `SSL_CERT_FILE`, and the README fix is part of #22.
- [A cached snapshot was built by an older package with other catalogue rules] → D3's format check against the recorded format, and later `Snapshot.catalog_format` (#23).
- [`main` or `latest` carries unreleased skill text, and skills are instructions to the agent] → opt-in only. `bundled` stays the default and the docs recommend `stable`. Snapshot content is never imported or run by rf-mcp or the package.
- [Snapshots pile up in the package's cache, about 1 MB per commit] → accepted for now. It's the package's cache, shared with its installer, and pruning is a non-goal.
- [The state file names a folder that a user deleted] → the startup check of `catalog.json` falls back to the wheel and reports it. The next refresh or `update-skills` restores it.
- [A long-running HTTP server never picks up new content] → documented, with `update-skills` plus a restart.

## Migration Plan

Nothing to migrate: the default `bundled` keeps today's behaviour. To roll back, remove `RF_MCP_OFFICIAL_SKILLS` (or `--official-skills`) from the agent entry. The state file and snapshots can be deleted at any time.

## Open Questions

- Whether `rf-mcp init` should suggest `--official-skills stable` once #21 has created `stable`. This only changes the docs or a default printout, not the mechanism.
