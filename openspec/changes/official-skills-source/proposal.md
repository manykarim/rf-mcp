# Proposal

## Why

rf-mcp serves the official Robot Framework skills from the `robotframework-plugins` wheel it depends on, so their content is only as new as the last PyPI release. Three days after `2026.11.0.dev1`, the repository's `main` was already 9 commits and 31 plugin files ahead. Users who want current skills have no way to get them except cloning the repository and passing it as `--skills-dir`. The wheel already contains a safe way to get newer content: `bundle.fetch(ref)` returns a `Snapshot` of any branch, tag or commit of MarketSquare/robotframework-plugins, cached per commit, with the same `catalog()` / `skills_dirs()` API. rf-mcp should be able to use it without giving up the offline, deterministic default.

## What Changes

- New setting `--official-skills` / `RF_MCP_OFFICIAL_SKILLS`: `bundled` (the default, unchanged behaviour) or a ref of the repository (`stable`, `latest`, `main`, a tag or a commit SHA).
- With a ref, the server serves the newest snapshot of that ref already in the cache. It never uses the network while starting. Without a usable snapshot, it serves the wheel's content and says why in the log and in `rf-mcp doctor`.
- A background refresh fetches the ref at most once per interval (default 24 hours, `RF_MCP_OFFICIAL_SKILLS_REFRESH`; `0` turns it off). New content is used from the next start. A failed fetch changes nothing.
- New command `rf-mcp update-skills [--ref REF]` fetches a ref on demand, for CI, Docker images and seeding the cache of an offline machine.
- `rf-mcp init --official-skills REF` writes the setting into the agent entries it generates.
- `rf-mcp doctor` reports where the official skills come from: the wheel, or a snapshot with its ref, commit and fetch time.
- The docs describe the channels, the offline fallback, and the existing alternative of a git clone as `--skills-dir`.

## Capabilities

### New Capabilities
- `official-skills-channel`: which content of the official skills rf-mcp serves (the wheel or a snapshot at a ref of the repository), how a snapshot is kept current, and how users choose, fetch and inspect it.

### Modified Capabilities
- `mcp-skills-provider`: "Official skills package is read through its public API" now covers snapshots fetched with the package's `fetch()` as well as its bundled content, with the same catalogue-format check.

## Impact

- **Code:** `src/robotmcp/skills/official.py` (choose the source; snapshot state), `src/robotmcp/config.py` (two settings), `src/robotmcp/skills/catalog.py` and `provider.py` (pass the config to the loader), `src/robotmcp/server.py` (start the background refresh), `src/robotmcp/onboarding/cli.py` and `agent_config.py` (`update-skills`, `init --official-skills`, doctor).
- **Files on disk:** snapshots live in the package's own cache (`%LOCALAPPDATA%\robotframework-plugins\snapshots` or `~/.cache/robotframework-plugins/snapshots`). rf-mcp adds one small state file recording which commit it fetched for each ref and when.
- **Network:** only with a ref set or `update-skills`: `api.github.com` and `codeload.github.com`, through the package. Mirrors and tokens depend on MarketSquare/robotframework-plugins#22.
- **Dependencies:** none new. Still `robotframework-plugins>=2026.11.0.dev1,<2027`.
- **Upstream:** `stable` doesn't exist until MarketSquare/robotframework-plugins#21 is resolved. Snapshots don't record their catalogue format until #23.
