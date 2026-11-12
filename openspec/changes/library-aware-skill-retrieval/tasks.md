# Tasks

## 1. Upstream and test fixtures

- [x] 1.1 Add a test fixture that installs a fake `robotframework_plugins.bundle` module in `sys.modules`, with `CATALOG_FORMAT`, `catalog()` and `skills_dirs()` over a `tmp_path` tree shaped like the official plugins (rf-web with `rf-browser` and `rf-selenium`, rf-api with `rf-requests`). Also add a fixture that builds a `--skills-dir` plugin repository with `plugins/<p>/catalog.yaml`. Verify that both are importable from `tests/conftest.py` and used by one smoke test.
- [ ] 1.2 Once the user agrees to post it, open an issue on MarketSquare/robotframework-plugins proposing an optional `libraries[].skills` field in `catalog.yaml` and `catalog.json`, which names the skills for each library (for example Browser → `rf-browser`). Verify that the issue URL is recorded in this file next to the task.

## 2. Catalogues and matching (D1)

- [x] 2.1 Create `src/robotmcp/skills/catalog.py` with `CatalogEntry` / `LibraryRef`, a reader for each source (official bundle, `--skills-dir` `plugins/*/catalog.yaml`, built-in) and `match(inventory, entries, served) -> list[Match]` with PEP 503 names, `SpecifierSet` ranges, the fallback to the plugin's skills, and ordering by layer. Verify with unit tests for every scenario of the "Skill catalogues match installed libraries" requirement, including an invalid `catalog.yaml`, which is skipped with a log warning.
- [x] 2.2 Add `src/robotmcp/skills/builtin/catalog.yaml`: Browser, SeleniumLibrary, AppiumLibrary and PlatynUI map to `rf-locators`, and RequestsLibrary maps to `rf-requests`. Make sure the wheel includes it. Verify with a test that reads it from `BUILTIN_SKILLS_ROOT` and with `uv build` plus a listing of the wheel.

## 3. Official skills layer (D2)

- [x] 3.1 Import `robotframework_plugins.bundle` once at startup, accepting only `CATALOG_FORMAT` in `{1}`. Insert `skills_dirs()` into `resolve_skill_roots` between `--skills-dir` and built-in. Report the package version, its format and the number of skills in `rf-mcp doctor`, with a warning on an unsupported format. Verify with tests for the new scenarios of the modified "Layered skill roots" requirement (official served, official over built-in `rf-requests`, `--skills-dir` over official, unsupported format) using the fake bundle, and run the existing `tests/test_skills_provider.py` unchanged.
- [x] 3.2 Document the official layer and the new precedence in `docs/CONFIGURATION.md` (Skills) and `README.md`. Verify with `grep -n "robotframework-plugins" docs/CONFIGURATION.md README.md`, and check that the precedence list matches the spec.

## 4. Project library inventory (D3)

- [x] 4.1 Add `EnvironmentResolver.installed_distributions(root)`, which probes the project interpreter (or the `python` next to the robotcode executable), keys names by canonical name, is cached under the environment fingerprint, and returns `None` for the fallback environment and on probe failure. Verify with unit tests using an injected probe: the scenarios "Project virtual environment", "Package installed later" (fingerprint change) and "Libraries not visible". Also add one test against this repository's real `.venv`, which must list `robotcode`.

## 5. Pointers on the first result (D4, D7)

- [x] 5.1 Add `ServerConfig.skill_pointers` (`RF_MCP_SKILL_POINTERS`, default on) and the `SkillPointers` middleware, registered after `ClientRoots`. It runs the inventory and the match in a worker thread alongside the call, adds `project_skills` (at most 3) to the first `rf_*` result per session and root in server mode, and rewrites the structured content and the JSON text block consistently. Verify with in-memory `Client` tests for every scenario of "Project skills on the first result", including another root in the same session and client mode. Also test that `RF_MCP_SKILL_POINTERS=0` adds nothing and that a middleware exception returns the original result.
- [x] 5.2 Add the `rf_mcp.skills.pointed` counter (attributes `skill` and `source`) to `src/robotmcp/telemetry.py` and record it from the middleware. Verify with an in-memory metric reader test for the "Pointer counted" scenario, and document the metric next to `rf_mcp.skills.read` in the docs.
- [x] 5.3 Measure the first-call overhead. Time `rf_discover action=info` on this repository with pointers on and off, over 10 runs each. Verify that the median difference is under 150 ms and record the numbers in this file next to the task.
  - 2026-11-12, this repository (rf-mcp's own `.venv`, 116 packages), fresh server per run: pointers off median 1115 ms, on 1117 ms, difference **2 ms** over 10 runs each. The package listing alone takes about 220 ms (5 runs), which the robotcode run hides. A call that starts no robotcode, such as `rf_discover action=read`, would show it once per session and root.

## 6. Hint links (D5)

- [x] 6.1 Add `skills: list[str]` to `HintRule` and `Hint`, loaded from `hints.toml` and validated (each entry must start with `skill://`, and invalid entries are reported like other rule errors). In the `SkillPointers` middleware, resolve each hint's candidates to `skill` (first served) for `rf_*` results and `rf_diagnose`, and count each one as `source=hint`. Verify with tests for the scenarios "Official skill served", "Official skills absent", "Project rule links a company skill" and "Linked skill not served".
- [x] 6.2 Add links to the built-in rules for Browser (strict mode, locator timeout, click intercepted, role prefix, invalid selector), iframes (`frame-not-found`), SeleniumLibrary (`selenium-element-interactable`, `element-stale`), Appium (`mobile-context-mismatch`, `appium-locator-or-source`) and RequestsLibrary (`requests-named-args`, `rf-dict-bracket-access`). Each names the official reference first and the built-in one second. Verify with the "Built-in links stay valid" test against the built-in folder and a pinned official repository snapshot. Skip that test with a reason when the snapshot isn't available.
  - Done with one deviation. `frame-not-found` links only the built-in `rf-locators/references/shadow-dom-iframes.md`, which covers Browser and SeleniumLibrary, and `element-stale` and `locator-invalid-selector` get no link. These rules fire for both libraries, and "served" doesn't mean "used by this project", so an official link would send a Selenium project to Browser docs. Eleven rules carry links. The snapshot test reads `RF_MCP_OFFICIAL_PLUGINS_DIR` (a checkout at 37562367), and all 21 links resolve.
- [x] 6.3 Document `skills = [...]` in the `hints.toml` section of `docs/CONFIGURATION.md`, with the company-skill example from the spec. Verify that the documented example loads in a test.

## 7. Instructions (D6)

- [x] 7.1 Pass the skill layers to `build_server_instructions`. For the official layer, omit the per-skill rows, and add the one catalogue line and the command-to-tool mapping line. Verify with tests for "Official skills served", "No official skills" and "Catalogue with the official package": `rf-robotcode` is listed and the length is at most 2,500 with the fake bundle holding nine skills plus two project skills.

## 8. Evaluation (D8)

- [x] 8.1 Build `experiments/library-aware-skills-eval/`: an OpenRouter tool-calling loop over an in-memory `create_server`, the Browser and RequestsLibrary tasks on the cookbook fixtures, the arms A/B/C, and graders (rf_analyze clean, rf_run pass). It reads `OPENROUTER_*` from `.env` and never prints it, and `.env` stays untracked. Verify with one dry run per arm on one model, where every arm completes and is graded.
  - 2026-11-12, `minimax/minimax-m3.1-flash`: every arm of both tasks completed and was graded, and all six passed, with each passing suite also failing against the broken copy of the app. The dry runs fixed four things:
    - the agent got `write_file` / `read_file` tools, as a coding agent has them;
    - the evaluation environment moved to a short path, because Chromium did not unpack under Windows' path limit;
    - the card mutation reports three digits, because the first and last four digits of 4242… are the same;
    - the turn limit is now 60.
  - They also showed that the real catalogue needed the skill-name rule in D1.
- [x] 8.2 Run 3 small models × 2 tasks × 3 arms × 5 repetitions against an official repository clone at a pinned commit. Write `openspec/changes/library-aware-skill-retrieval/eval-report.md` with pass rates, median turns and tokens, pointer follow-through (`skills.read` after `skills.pointed`), the gate verdict from D8, and the content findings for upstream. Verify that the report exists and names the commit, the models and the run count.
  - Done: `eval-report.md` has 90 runs at 37562367 and cost $1.23. **The gate fails.** B passed 0.97 against A's 1.00, a difference of one run, and used 17% more tokens, from reading `rf-language` and both locator skills.

## 9. Required dependency

- [ ] 9.1 Blocked on the PyPI release of MarketSquare/robotframework-plugins#5. Add `robotframework-plugins>=2026.11,<2027` to the dependencies in `pyproject.toml`, refresh `uv.lock`, replace the fake bundle in one integration test with the real package, and update the README install notes. Verify that `uv run pytest -q` passes, and that `uv tool install .` in a clean tool environment serves `skill://rf-browser/SKILL.md`.

## 10. Integration

- [ ] 10.1 Run `uv run pytest -q` on Python 3.10 and 3.12, ruff on `src tests`, and `openspec validate library-aware-skill-retrieval --strict`. Verify that all are clean apart from the ruff findings already on `main`, and that the CI matrix passes on the PR.
  - Local part done on 2026-11-12:
    - pytest passes on Python 3.10 (368 passed), 3.12 (368) and 3.13 (368, plus the 3 Browser cookbook suites that fail only on this machine's stale `.venv` Browser folder);
    - the cookbook passes against an interpreter with a working Browser (6 passed);
    - ruff shows only the 8 E741 findings in `tests/fixtures/ledger` that are already on `main`;
    - `openspec validate --strict` is clean.
  - Open until the PR's CI runs.

## Workflow follow-up

- Archive `project-skills-follow-client-roots` before this change. Both modify "Layered skill roots for ecosystem plugins and project customizations", and this delta builds on that one's wording.
- Mark `bundle-agent-skills` as superseded by this change, or drop it.
- If the gate in 8.2 passes, propose the follow-up change that retires the built-in `rf-locators` / `rf-requests` library content and the "Verified library cookbooks" requirement.
- File a second upstream issue for reading the project install record (`.rf-agentskills/installed.json`), so skills copied in by `robotframework-plugins install` are not served twice.
- Archive this change after the PR is merged.
