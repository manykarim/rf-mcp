# Design

## Context

The motivation is in proposal.md, and the behaviour is in the three spec deltas. This document covers the code they touch.

- **Skills.** `ProjectSkillsProvider` (`src/robotmcp/skills/provider.py`) builds the skill folder list on each request through `resolve_skill_roots(config, project_roots=...)`:
  1. project skills;
  2. `--skills-dir`, expanded so that a plugin repository contributes each `plugins/*/skills`;
  3. built-in.

  A `SkillsDirectoryProvider` is cached for each folder list, and the first name wins.
- **Hints.** `HintEngine.diagnose` runs inside `CliRunner.execute` and `SessionRunner`, and they don't know the MCP session or the served skills. Tools return plain dicts (`CommandResult.to_response()`), and hints are in them as `hints: [{id, title, message, fix_examples, relevance}]`.
- **Environment.** `EnvironmentResolver.resolve(root)` caches a `ProjectEnvironment` (`interpreter`, `source`, `libraries_visible`) under a fingerprint that includes the mtimes of the project's `site-packages`. A package install therefore invalidates it.
- **Session state.** `ClientRoots` is middleware that fills a missing `root` from the client's MCP roots, and caches the roots per session.
- **Official package.** `robotframework-plugins` has no public API yet. MarketSquare/robotframework-plugins#5 proposes `bundle.CATALOG_FORMAT`, `bundle.catalog()` and `bundle.skills_dirs()`, and the package isn't on PyPI yet.
  - The repository layout is already usable: `plugins/<p>/catalog.yaml` and `plugins/<p>/skills/<skill>/`.
  - Its catalogue entries map libraries to a plugin, not to a skill. rf-web ships both `rf-browser` and `rf-selenium`.
- **Prior evaluation.** The `bundle-agent-skills` evaluation found that whole skills loaded on top of rf-mcp gave no gain, or a loss when the content was wrong. Targeted, correct content is the case left to test.

## Goals / Non-Goals

**Goals:**
- One code path for every catalogue source: the official package, plugin repositories in `--skills-dir`, and the built-in catalogue.
- No extra round trip and no noticeable delay on the first call. Nothing at all on later calls.
- Work before #5 lands: through a `--skills-dir` clone, and against a fake `bundle` module in tests.
- A measured answer to "do pointers help?" before the built-in library skills are retired.

**Non-Goals:**
- Choosing between several skills inside one plugin by reading their text. If the catalogue doesn't say, only the skill names are compared with the library name (D1).
- Excerpts inlined into results, scanning suites for `Library` imports, and tracking `Import Library` in the REPL.
- Making rf-mcp's built-in catalogue describe the official plugins. That would copy upstream metadata.

## Decisions

### D1. One catalogue model, three readers

The new `src/robotmcp/skills/catalog.py` normalises every source into `CatalogEntry(plugin, layer, skills_dir, skills: list[str], libraries: list[LibraryRef(name, package, versions, skills)])`. The readers are:

- **Official package:** `bundle.catalog()` plus `bundle.skills_dirs()`, used only when `bundle.CATALOG_FORMAT` is in `SUPPORTED_FORMATS = {1}`.
- **`--skills-dir`:** for each configured folder that has `plugins/*/catalog.yaml`, parse it with `yaml.safe_load` (pyyaml already comes in through fastmcp), and take the skill names from `plugins/<p>/skills/*/SKILL.md`. Folders without a catalogue contribute skills but no matching.
- **Built-in:** `src/robotmcp/skills/builtin/catalog.yaml` maps Browser, SeleniumLibrary, AppiumLibrary and PlatynUI to `rf-locators`, and RequestsLibrary to `rf-requests`. It has no version ranges.

`libraries[].skills` is optional everywhere. When it's missing, the plugin's skills whose name contains the library's name without a `Library` suffix count (`rf-browser` for Browser, `rf-requests` for RequestsLibrary), or all of the plugin's skills when none does. The pinned official catalogue has no per-library skill list, and the first dry run of the evaluation showed what happens without this rule: a Browser project with RequestsLibrary installed got `rf-requests` and `rf-restinstance` ahead of `rf-browser`. The matches are also reordered so that each library's first skill comes before any library's second.

*Alternative:* Read the official package through `importlib.resources` and its private `_assets`. That was rejected: it breaks silently on upstream refactors, and #5 exists to avoid exactly that.

### D2. The official layer sits in `resolve_skill_roots`

`resolve_skill_roots` inserts the official `skills_dirs()` values after `--skills-dir` and before `BUILTIN_SKILLS_ROOT`, without duplicates. Precedence comes from `SkillsDirectoryProvider`'s first-name-wins rule, as today.

The bundle is imported once at startup. A missing package, or an unsupported format, gives an empty layer. The format case also logs a warning and becomes a `doctor` line.

### D3. Inventory: one probe per environment, run alongside the first call

`EnvironmentResolver` gets `installed_distributions(root) -> dict[str, str] | None`. It keys names by `packaging.utils.canonicalize_name`, and is cached under the same fingerprint as `resolve`.

The probe runs the project's interpreter once:

```
python -c "import importlib.metadata as m, json; print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
```

It uses the same timeout and `stdin=DEVNULL` handling as `probe_robotcode_version`. The interpreter is `env.interpreter`, or else the `python` next to `command_prefix[0]` (`_executable_next_to`).

There's no probe when `libraries_visible` is false. A timeout, a non-zero exit or invalid JSON returns `None` and logs at info level.

*Alternative:* Read `site-packages/*.dist-info` directly. It's faster, but it misses `.pth`/editable installs and the uv layered environments. The subprocess is what the project interpreter itself sees.

### D4. A middleware adds `project_skills` and resolves hint links

The new `SkillPointers` middleware is registered after `ClientRoots`, so `arguments["root"]` is already filled in. In `on_call_tool` for `rf_*` tools in server mode, it:

1. Resolves the call's root the way the tools do (`cfg.resolve_root(arguments.get("root"))`).
2. If `(session id, root)` is unseen, starts the inventory and the match in a worker thread (`anyio.to_thread`) before `call_next`, and awaits it afterwards. That overlaps the probe with the tool's own robotcode run.
3. After `call_next`, rewrites the result's structured content and its JSON text block:
   - It adds `project_skills` (at most three entries, ordered by layer, then catalogue order, then library order; one entry per library and skill) and marks the pair as seen. An empty match adds nothing, and the pair stays seen, so the probe isn't repeated.
   - It replaces each hint's `skills` candidate list with `skill`, the first served candidate, or removes it.
   - It records `rf_mcp.skills.pointed` for each pointer it emits.

The session key is the MCP session id for clients of the handshake-era protocol, whose `session_id` stays the same across calls (unlike `id(ctx.session)`, which changes on every call). Clients of the stateless 2026-07-28 protocol have no session that lasts across calls, so they share one key for the server process. For stdio, that is the single client.

"Served" means a skill name in `discover_skill_summaries(resolve_skill_roots(cfg, project_roots=[root]))`, plus the file existing for a `references/...` URI. This is cached for each folder tuple and checked again on the next request, so edits show up.

A tool error result passes through untouched. A failure inside the middleware is logged and the original result is returned unchanged.

*Alternative:* Add the field in each tool function. That's seven call sites, and the runners have no session context. A middleware also covers `rf_diagnose` and client mode (hint links only) in one place.

### D5. Hint rules carry candidate links, the server resolves them

`HintRule` gets `skills: list[str] = []`, which `hints.toml` validates and loads like the other fields, and `Hint` gets the same field. `to_hint()` copies it.

The engine stays unaware of served skills. Resolving the links is D4's job, because only the server knows the root's skill layers. Built-in rules gain links for:
- Browser locator and strict-mode errors;
- iframes;
- SeleniumLibrary element errors;
- Appium contexts;
- RequestsLibrary named arguments and dictionaries.

Each link names the official reference first, then the built-in one. A unit test walks every built-in link against the pinned official package and the built-in folder (spec "Built-in links stay valid").

### D6. Instructions

`build_server_instructions` receives the layer of each skill summary. For the official layer, it:
- omits those rows from the catalogue and adds one line: "Official library skills (robotframework-plugins): list_resources; the ones matching your project's libraries come with your first tool result.";
- adds one mapping line before the catalogue: "These skills show robotcode commands; use rf_repl for robotcode repl, rf_run for robot, rf_discover, rf_analyze, rf_results, rf_doc for doc/libdoc, rf_debug for debug."

Both lines together are about 330 characters. The existing 2,500-character cut-off stays, and a test checks the length with the official package present.

### D7. Pointers can be switched off for the evaluation

`ServerConfig.skill_pointers: bool = True`, read from `RF_MCP_SKILL_POINTERS`, turns off both `project_skills` and hint `skill` resolution. It exists so the evaluation can compare arms on the same build. It's documented as a troubleshooting switch, not a feature.

### D8. Evaluation design

The harness lives in `experiments/library-aware-skills-eval/` and isn't packaged. It's a small agent loop: an OpenRouter chat model with tool calling, connected to an in-memory `create_server(...)` through `fastmcp.Client`. `OPENROUTER_*` comes from `.env`, and the harness never logs or commits it.

- **Models:** three small, cheap ones.
- **Tasks:** they run against the existing cookbook fixtures. A Browser task needs a strict-mode fix and an iframe `>>>`, and a RequestsLibrary task has a nested payload and a negative check.
- **Arms:**
  - A: built-in only, with no official skills folder;
  - B: an official repository clone at a pinned commit through `--skills-dir`, with pointers;
  - C: as B, but `RF_MCP_SKILL_POINTERS=0`.
- **Size:** five repetitions per cell.
- **Metrics:** pass (rf_analyze clean and rf_run passing), turns, tokens, and `skills.pointed` against `skills.read` from an in-memory metric reader.

**Gate for the follow-up change** that retires the built-in library skills: B's pass rate is at least A's, and B's median tokens are at most 1.1 × A's. A fail is reported upstream as content findings. It isn't a reason to undo this change.

## Risks / Trade-offs

- [The first call gets slower by the probe (about 100–300 ms) when the tool itself is faster than that] → It runs in parallel with the tool, once per session and root, and is cached across sessions until the environment changes.
- [Official skills teach `robotcode` commands, and agents follow them around rf-mcp] → The D6 mapping line. The evaluation's arm B measures it, and if it still happens, the next step is an upstream "with rf-mcp" section.
- [A plugin-level match points to the wrong skill (Browser → `rf-selenium`)] → Upstream `libraries[].skills` (task 1.2). Until then, at most three entries keep the noise bounded, and the spec states the fallback.
- [Official `rf-requests` shadows the built-in `rf-requests` as soon as the package is installed] → This is intended under the chosen direction. The evaluation's RequestsLibrary task covers it, and the built-in copy stays as the fallback without the package.
- [Rewriting a tool result's text block can break clients that parse it] → The JSON is re-serialised from the same structured content, and a test compares both forms for every `rf_*` tool.
- [Upstream changes the catalogue] → `CATALOG_FORMAT` check, a pinned range `>=2026.11,<2027`, and a `doctor` warning.
- [Prior evidence says skills on top of rf-mcp don't help] → Pointers are targeted, and the gate in D8 decides before any built-in content is removed.

## Migration Plan

1. Ship with the official layer working when the package can be imported. Without it, nothing changes except the hint links to built-in references and `project_skills` from the built-in catalogue.
2. When #5 is released, declare `robotframework-plugins>=2026.11,<2027` as a required dependency (task 9.1), and update the README install notes.
3. **Rollback.** Set `RF_MCP_SKILL_POINTERS=0` to stop the pointers. Uninstalling the package, or pinning rf-mcp to the earlier release, removes the official layer.

## Open Questions

- The exact wording of the instruction lines, to be tuned during the evaluation within the character budget.
- Whether `project_skills` also belongs in `rf-mcp doctor` output, as a human-readable listing. It's cheap to add later without a spec change.
