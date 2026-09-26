# rf-mcp 0.36.0

Install fixes, project-keyword discovery, and a sturdier `intent_action` for smaller models.
Includes the unreleased 0.35.1 install-blocker fix.

## ⚠️ Install: `rf-mcp[all]` works again with uv / pipx

In 0.34.0 and 0.35.0, `uv tool install "rf-mcp[all]"` — the README quick start — failed
for every uv/uvx/pipx user. `desktop` is no longer part of `[all]`; install it explicitly:

```bash
uv tool install "rf-mcp[all]"                           # web, API, mobile, DB, memory
uv tool install --prerelease=allow "rf-mcp[desktop]"    # PlatynUI desktop (opt-in)
pip install "rf-mcp[desktop]"                           # pip/poetry/pdm need no flag
```

**Migration:** if you relied on `[all]` for desktop automation, add `rf-mcp[desktop]`.

## Highlights

- **Project keywords work in every tool.** Keywords from your own libraries and resource
  files are found by `find_keywords` (also in web sessions) and documented by
  `get_keyword_info` — including qualified names (`MyLib.My Keyword`), resource and alias
  scoping, and `mode="library"` for resources. `get_session_state` lists them under
  `project_sources`.
- **Generated suites replay what ran.** `build_test_suite` imports project libraries by path
  with their arguments and alias, and keeps a `Library.` prefix when it disambiguates a
  keyword — so `run_test_suite` passes suites that passed live.
- **`intent_action` is recoverable for weaker models.** A literal `"null"`/`"None"` is treated
  as absent; a missing `target` is rejected naming the parameter with an example; an omitted
  `session_id` resolves to the only session with libraries loaded (reported as `session_note`)
  instead of silently using an empty `default` session.
- **Installer safety.** `--dry-run` no longer installs; declining the confirmation prompt is
  honoured; failures exit non-zero; `uninstall -C` targets the right project; JSONC configs are
  no longer corrupted by trailing-comma handling.
- **Launch fidelity.** Project-aware launches keep the extras you installed (Browser /
  SeleniumLibrary no longer go missing) and no longer shadow the project's Robot Framework.

## Fixes

- A library imported with arguments or an alias no longer gains a second instance with
  default configuration before each step.
- `set_library_search_order` accepts project libraries and reports entries it cannot apply
  instead of dropping them silently.
- `check_library_availability` accepts library and resource file paths.
- `execute_flow` steps accept `args` as well as `arguments`, like `execute_batch`.
- Seven latent crashes on error and fallback paths, e.g. a `NameError` at import time when
  Robot Framework is unavailable, and snapshot compression (`fold_lists`) failing outright.
- Dashboard (frontend) validation fixes: honest browser/platform metadata, session-switch race.

## Known limitation

All sessions in one rf-mcp process share a single Robot Framework namespace: two sessions
importing same-named project libraries or resources can collide. Use one project per server.

## Dependencies

Notable upgrades: Django 4.2 → **5.2**, cryptography 46 → **50**, transformers 4 → **5**
(`semantic` extra), pydantic-ai-slim 1.37 → 1.106, requests 2.33, urllib3 2.7, lxml 6.1,
pyjwt 2.14 (clears 7 advisories).

## Docs

- `docs/rf-mcp-0.36.0-libdoc.{html,json}` — MCP tool reference (57 tools, 19 enabled by
  default), now including each tool's parameter schema.
- `docs/robotmcp.html` — Robot Framework libdoc for `McpAttach`.
- Regenerate both with `uv run invoke libdoc`; `uv run invoke release` also builds dist/.
