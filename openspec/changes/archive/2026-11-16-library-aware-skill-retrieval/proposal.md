# Proposal

## Why

The official Robot Framework skills now live in `MarketSquare/robotframework-plugins`, published as the `robotframework-plugins` package. If rf-mcp keeps its own library skills next to them, the same advice is maintained twice and drifts apart. If rf-mcp only serves files, it adds nothing an agent can't read from a skills folder.

rf-mcp knows things a skills folder can't:
- which libraries and versions a project has installed;
- which keyword just failed, and with what error.

So it can hand the agent the right section of the official skills at the moment it matters, rather than hoping the agent reads the right skill up front.

The 2026-08 evaluation for `bundle-agent-skills` showed why this has to be targeted. A whole skill loaded on top of rf-mcp added nothing or did harm (one wrong iframe combinator displaced rf-mcp's correct guidance). The value showed up only where the content was correct and actually needed.

## What Changes

- **Official skills as a layer.** When `robotframework-plugins` is installed, rf-mcp serves its bundled skills through the public `robotframework_plugins.bundle` API (MarketSquare/robotframework-plugins#5). The serving order becomes:
  1. project skills;
  2. `--skills-dir`;
  3. official skills;
  4. rf-mcp's built-in skills.

  A `--skills-dir` that holds a plugin repository with `plugins/*/catalog.yaml`, such as a clone or a company mirror, gets the same library matching.
- **Project library inventory.** For each project root, rf-mcp lists the distributions installed in the project's environment, with their versions. It caches the list with the environment it already resolves.
- **Matching through the catalogue.** rf-mcp matches the installed packages against the catalogue entries `libraries[{name, package, versions}]`, from the official package and from `--skills-dir` plugin repositories. It also reads a small catalogue for its own built-in skills. Each match names the skills to read and says whether the installed version is in the supported range.
- **Project skills on the first result.** In server mode, the first tool result of a session for a project root carries a short `project_skills` list. For example: Browser 20.1.0 → `skill://rf-browser/SKILL.md`, supported. It is capped at three entries.
- **Hints link to skill sections.** A hint rule can name skill URIs in order of preference. The attached hint carries the first one the server serves. For example, a strict-mode violation links `skill://rf-browser/references/locators.md`, and falls back to `skill://rf-locators/references/browser.md` when the official skills are absent. `hints.toml` rules can link a company's own skills the same way.
- **Tool vocabulary note.** When official skills are served, the server instructions map their `robotcode` commands to rf-mcp tools (`robotcode repl` → `rf_repl`, and so on). The skill catalogue in the instructions lists the official skills as one line instead of one row each, so `rf-robotcode` and the project skills are never cut off by the 2,500-character limit.
- **Measurement.** A new counter `rf_mcp.skills.pointed` (attributes: skill, how it was pointed to) sits next to `rf_mcp.skills.read`. An OpenRouter evaluation on small models compares three setups: built-in only, official skills with pointers, and official skills without pointers.
- **Required dependency.** Once `robotframework-plugins` with the `bundle` API is on PyPI, rf-mcp declares it as a required dependency (`>=2026.11,<2027`). Until then the layer is used whenever the package can be imported.

Not in this change:
- Removing the built-in library skills (`rf-locators`, `rf-requests`). That's a follow-up change, decided by the evaluation result.
- Excerpts pasted into results instead of pointers.
- Scanning suites for `Library` imports.
- Detecting skills that `robotframework-plugins install` already copied into the agent's folders. That needs a second upstream issue first.

## Capabilities

### New Capabilities
- `library-aware-skill-retrieval`: the project library inventory, matching it against skill catalogues, and pointing the agent to matching skills on its first result and from hints.

### Modified Capabilities
- `mcp-skills-provider`:
  - The layered skill roots gain the official-skills layer, between `--skills-dir` and the built-in skills.
  - The server instructions gain the tool vocabulary note, and list the official skills as one catalogue line.
- `robotcode-error-hinting`: hint rules, built-in and from `hints.toml`, can link skill sections, and an attached hint carries the first served link.

## Impact

- **Code:**
  - `src/robotmcp/skills/provider.py`: official layer, catalogue reading, instructions.
  - New `src/robotmcp/skills/catalog.py`: catalogues and matching.
  - `src/robotmcp/runner/environment.py`: probing installed distributions, cached per root.
  - `src/robotmcp/server.py`: the first-result `project_skills` field.
  - `src/robotmcp/hints/`: the `skills` field on rules, `Hint.skill`, and links on the built-in rules.
  - `src/robotmcp/telemetry.py`: the new counter.
  - `src/robotmcp/skills/builtin/catalog.yaml`: new built-in catalogue.
- **Dependencies:** `robotframework-plugins` becomes required after #5 is released. `pyyaml` is already present through fastmcp.
- **Responses:** additive fields (`project_skills`, `hint.skill`). No existing field changes.
- **In-flight changes:**
  - This change modifies the same requirement as `project-skills-follow-client-roots`, and builds on its wording. Archive that one first.
  - It supersedes the stopped `bundle-agent-skills` change.
- **Docs:** `docs/CONFIGURATION.md` (Skills) and `README.md`.
