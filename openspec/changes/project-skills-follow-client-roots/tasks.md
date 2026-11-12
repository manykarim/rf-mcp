# Tasks

## 1. Shared roots lookup

- [x] 1.1 Add a public `async roots_for(ctx) -> list[Path]` to `ClientRoots` that wraps the existing per-session cached lookup, and make `on_call_tool` use it (D2). Verify that the roots tests in `tests/test_server_tools.py` still pass unchanged.

## 2. Per-request skill roots

- [x] 2.1 Give `resolve_skill_roots` a `project_roots` argument: project skill folders for each root in order, then `--skills-dir` folders, then the built-in folder. Without the argument it keeps today's startup behaviour (D2). Verify with unit tests in `tests/test_skills_provider.py` that cover one root, two roots and the pinned root.
- [x] 2.2 Implement `ProjectSkillsProvider` (D1). It works out the folder list per request: the pinned root, else the client's roots in server mode via `roots_for`, else `startup_root()`, falling back when there is no request context. It delegates the four resource hooks to a `SkillsDirectoryProvider(reload=True)` cached per folder list. Mount it in `mount_skills_provider`, and pass the `ClientRoots` instance from `server.py`. Verify with in-memory `Client(..., mode="legacy")` tests:
  - a server started in an empty folder with a client root that holds `.robotframework/skills/shop-conventions/` lists and reads `skill://shop-conventions/SKILL.md`, also through the `list_resources` / `read_resource` tools;
  - two roots serve both projects' skills, and the first root wins a duplicate name;
  - `--root` ignores the client roots;
  - a project skill named `rf-locators` wins over the built-in one;
  - after `notifications/roots/list_changed` the next listing uses the new roots;
  - two concurrent clients with different roots each see only their own project skills.
- [x] 2.3 Update the "How the project root is found" section of `docs/CONFIGURATION.md`: project skills follow the call's root, and the VS Code note about user entries goes. Update the precedence list under "Skills" if its wording differs from the spec. Verify that the docs say nothing anywhere about project skills being read only at startup (`grep -n "once at" docs/`).

## 3. Instructions note

- [x] 3.1 Add the line from D3 to `build_server_instructions` when no root is pinned and `startup_root()` reports source `cwd`. Verify with tests: the line is present for a server started in an empty `tmp_path`, absent for one started in a folder with `robot.toml`, and the instructions stay at most 2,500 characters with the full built-in catalog.

## 4. Integration

- [ ] 4.1 Run `uv run pytest -q` on Python 3.10 and 3.12, ruff on `src tests`, and `openspec validate project-skills-follow-client-roots --strict`. Verify that all are clean apart from the ruff findings already on `main`, and that the CI matrix passes on the PR.
- [ ] 4.2 Repeat the VS Code check from task 6.4 of `project-independent-server` with the real server. Register the branch's `rf-mcp` as a user entry in `~/.copilot/mcp-config.json`, open a project that has `.robotframework/skills/<name>/SKILL.md`, and ask the agent to list its skills, which requires the user in the VS Code GUI. Verify that the project skill is listed, then remove the entry again.

## Workflow follow-up

- Archive the change after the PR is merged.
- Review `bundle-agent-skills` against the official skills package announced for MarketSquare/robotframework-plugins.
