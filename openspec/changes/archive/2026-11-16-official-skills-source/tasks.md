# Tasks

## 1. Settings and test fixtures

- [x] 1.1 Add `ServerConfig.official_skills` (`--official-skills` / `RF_MCP_OFFICIAL_SKILLS`, default `bundled`, the option winning over the environment) and `ServerConfig.official_skills_refresh_hours` (`RF_MCP_OFFICIAL_SKILLS_REFRESH`, default 24, `0` off, an invalid value falling back to 24 with a log warning). Verify with config tests for the three scenarios of "Official skills channel setting" and for "Refresh off".
- [x] 1.2 Extend the fake bundle in `tests/conftest.py` with `Snapshot`, `FetchError` and a `fetch(ref, *, cache_dir=None, timeout=30.0)`. The fake writes a snapshot folder (`plugins/<p>/skills/...` plus `catalog.json`) for a configurable commit from the same plugin specs, or raises `FetchError` for refs set to fail, and counts its calls. Add a fixture that points rf-mcp's cache base at `tmp_path`. Verify with one smoke test that fetches a fake ref and reads `Snapshot.skills_dirs()`.

## 2. Snapshot state and the loader (D2, D3, D6)

- [x] 2.1 Add the state file `<cache base>/rf-mcp/official-skills.json` in `src/robotmcp/skills/official.py`: read it (missing or invalid means empty), record a fetch (`commit`, `path`, `fetched_at`, `checked_at`, `catalog_format`, `package_version`), record a check, and write it atomically with a temporary file and `os.replace`. Verify with unit tests: a roundtrip, an invalid file read as empty, and two writers not corrupting the file.
- [x] 2.2 Make `load_official_skills(channel)` choose the source and pass `config.official_skills` from `resolve_skill_roots`, `build_server_instructions` and `catalog_entries`. `bundled` keeps today's behaviour. A ref with a recorded, complete snapshot in a supported, matching catalogue format serves that snapshot. Anything else falls back to the bundled content with `fallback_reason` and a log warning naming `rf-mcp update-skills --ref <ref>`. A full SHA is looked up the same way. Verify with tests for every scenario of "Startup serves a cached snapshot or the bundled content" and the new scenario of the modified "Official skills package is read through its public API". Assert that the fake `fetch` is never called during `create_server`. Run `tests/test_skills_provider.py`, `tests/test_skill_catalog.py` and `tests/test_skill_pointers.py` unchanged.
- [x] 2.3 Check that a served snapshot behaves like the bundled layer end to end: `list_resources` shows its skills, `project_skills` matches against its catalogue, and hint links resolve to its files. Verify with an in-memory `Client` test on a fake snapshot whose `rf-browser` differs from the bundled one, asserting that the snapshot's text is served.

## 3. Refresh and update-skills (D4, D7)

- [x] 3.1 Implement the refresh step: skip for `bundled`, a full SHA, an interval of `0` or a recent `checked_at`; otherwise record the check, then `bundle.fetch(ref)` and record the result, logging one warning on failure and keeping the old entry. Start it as a daemon thread from `robotmcp.entry.main` after `create_server`, never from `create_server` itself. Verify with tests for every scenario of "Background refresh of a ref channel" (due, not due, offline), driving the step directly with the fake `fetch`. Add a test that `main()` with a ref channel starts the thread without waiting for it.
- [x] 3.2 Add `rf-mcp update-skills [--ref REF]` to `src/robotmcp/onboarding/cli.py`, using the same fetch-and-record step without the interval. Print JSON with `ref`, `commit`, `path` and `skills`. Exit 1 on `FetchError`, or with channel `bundled` and no `--ref`. Verify with CLI tests for both scenarios of "update-skills fetches a ref on demand", plus a test that a following server start serves the fetched snapshot with no `fetch` call.

## 4. init and doctor (D7)

- [x] 4.1 Add `--official-skills CHANNEL` to `rf-mcp init`, adding `RF_MCP_OFFICIAL_SKILLS` to the environment of every generated entry: the project files, and the `claude`, `codex`, `agy`, `copilot` and `opencode` user-scope commands and files. Verify with tests of `generate_agent_mcp_configs` for project and user scope (the scenario "User-level entry for the latest skills") and that without the option no entry sets it.
- [x] 4.2 Extend `rf-mcp doctor` (text and `--json`) with the channel and what is served: the package version for `bundled`; the ref, commit, `fetched_at`, snapshot folder and skill count for a snapshot; and the fallback reason with the `update-skills` command. Verify with tests for both scenarios of "doctor reports the official skills source", with an assertion that the fake `fetch` is not called.

## 5. Documentation

- [x] 5.1 Document the channels in `docs/CONFIGURATION.md`:
  - rows for `--official-skills` / `RF_MCP_OFFICIAL_SKILLS` and `RF_MCP_OFFICIAL_SKILLS_REFRESH` in the settings table;
  - a "Newer official skills" subsection in Skills, covering `bundled` as the default, `stable` as recommended once it exists (#21), `latest`/`main` as unreleased, startup without network, the refresh interval, "applies at the next start" (including long-running HTTP servers), `update-skills` for images and offline machines, `SSL_CERT_FILE` for proxies that re-sign TLS, and the git clone as `--skills-dir` alternative.

  Mention `init --official-skills` and `update-skills` in `README.md`. Verify that every command shown runs as written against the fake or real package, and that `grep -n "RF_MCP_OFFICIAL_SKILLS" docs/CONFIGURATION.md README.md` finds both files.

## 6. Integration

- [x] 6.1 Add a live test, run only with `RF_MCP_LIVE_FETCH=1`: `rf-mcp update-skills --ref latest` against GitHub, then a server built with channel `latest` serves the snapshot's skills and `project_skills` names `rf-browser` for a Browser inventory. Run it once locally and record the commit and timings next to this task.
  - 2026-11-16, `tests/test_official_package.py::test_live_update_skills_then_serve_the_snapshot`: `latest` = `56547f0e5799`, fetched in 1.4 s and 1.1 s (two runs, each into a fresh cache), 9 skills; the whole test took 3.5 s and 3.2 s. The documented commands (`init --official-skills latest`, `update-skills --ref latest`, `doctor` with `RF_MCP_OFFICIAL_SKILLS=latest`, and an unknown ref, which exits 1) gave the expected output against GitHub with `LOCALAPPDATA` pointing to a scratch folder.
- [x] 6.2 Run `uv run pytest -q` on Python 3.10 and 3.12, ruff on `src tests` and `openspec validate official-skills-source --strict`. Verify that all are clean apart from findings already on `main`, and that the CI matrix passes on the PR.
  - 2026-11-16: Python 3.10 and 3.12 (`uv run --isolated`): 396 passed, 31 skipped each, after updating `test_doctor_reports_the_official_package` for the report's new fields. ruff: only the 8 E741 findings in `tests/fixtures/ledger` already on `main`. `openspec validate --strict`: valid. CI on #152: all 17 checks passed; merged as `ec5f019`.

## Workflow follow-up

- When MarketSquare/robotframework-plugins#23 lands, read `Snapshot.catalog_format` instead of rf-mcp's own record (D3).
- When #21 creates `stable`, revisit the open question in design.md about suggesting it in `init`.
- Archive the change after the PR is merged.
