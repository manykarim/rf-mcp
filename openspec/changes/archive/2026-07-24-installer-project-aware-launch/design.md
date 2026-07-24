# Design — installer-project-aware-launch

## Problem, precisely

rf-mcp executes RF keywords **in its own process**: `rf_native_context_manager` calls
`namespace.import_library(name)` (robot's native `importlib`) against the running interpreter's
`sys.path`. So the launch command in the MCP config decides which RF libraries the server can
import. The installer's `resolved_command()` always emits rf-mcp's own shim → the server can only
see rf-mcp's bundled libraries, never the project's.

**Proven during exploration:**
- Project `.venv` with `robotframework-jsonlibrary` → rf-mcp's own Python: `ModuleNotFoundError:
  No module named 'JSONLibrary'`. `uv run --python <proj>/.venv/bin/python --with rf-mcp python -c
  "import JSONLibrary; import robotmcp"` → **both import**, `JSONLibrary` resolved from the project
  venv, zero mutation, cwd-independent.
- **uv `--with` overlay only layers the target's site-packages when the target is a real
  virtualenv (`pyvenv.cfg`).** Against `/usr/bin/python3` (no `pyvenv.cfg`) it builds an isolated
  env and the base libs stay invisible (`apt_pkg` not importable). → conda/global are the true
  non-uv fallbacks.
- **Dependency conflict:** a project pinning `robotframework==6.1.1`, overlaid with `uv run --with
  rf-mcp` → RF silently **upgraded to 7.4.2** (tests the wrong RF); `PYTHONPATH=<proj site-packages>`
  onto rf-mcp's Python → project RF 6.1.1 wins and can break rf-mcp (built for RF≥7); installing
  rf-mcp *into* a hard-pinned project → resolver fails. Only the **attach bridge** keeps both intact.
- `<abs sys.executable> -m robotmcp.server` == the console script for launching (server.py
  `__main__` → the same `main()` that runs `_protect_mcp_stdout()`), so `-m` carries no
  stdio-safety regression.

## The uv-first decision tree (the core)

Detection inputs: `project_dir` (from `-C/--project-dir`, else cwd), `project_python` (from env
markers), `uv_available`, `rfmcp_in_project` (`project_python -c "import robotmcp"`), `extra_libs`
(RF libs the project references that `rf-mcp[all]` does **not** bundle — from installed
site-packages **and** `Library`/`Resource` imports scanned in the project's `.robot`/`.resource`
files), `rf_conflict` (project pins RF<7 / a shared dep incompatible with rf-mcp / python<3.10).

```
① rfmcp_in_project                    → run IN the project env (best; no overlay, no mutation)
                                          uv project:  uv run --project <dir> robotmcp
                                          else:        <project_env>/bin/robotmcp
                                                       or  <project_python> -m robotmcp.server

② extra_libs & !rfmcp_in_project       → PRIMARY DEFAULT: uv overlay (non-mutating, cwd-independent)
   & uv_available & venv & !conflict      ALL venv types:  uv run --no-project --python <env-py> <with-spec> robotmcp

   IMPLEMENTATION CORRECTIONS (proven under test; the initial draft above was wrong):
   • Use `--no-project --python <env-py>` (uniform for plain/poetry/pdm/pipenv/rye/hatch),
     NOT `--project <dir>`: `--project` only syncs the project's DECLARED deps and misses
     libraries installed ad-hoc into the venv, whereas `--python <env-py>` layers the venv's
     ACTUAL site-packages and is cwd-independent (verified from an unrelated cwd).
   • `<with-spec>` is editable-aware: `--with-editable <src>` when the installed rf-mcp is a
     local/editable/dev build (its version may not be on PyPI, so a pin would fail to resolve),
     else `--with rf-mcp==<own-version>` for a published install, else unpinned `--with rf-mcp`.

③ no project env, OR extra_libs empty  → rf-mcp's OWN shim  (today's behaviour — correct here)
   (only libs rf-mcp[all] bundles)        guard: confirm rf-mcp's env actually has the needed extras

── fallbacks (uv can't overlay a non-venv prefix, or no uv) ──
④ conda / bare-global                  → co-install rf-mcp into that env, run it there
                                          conda run -n <env> robotmcp  /  <prefix>/bin/robotmcp
⑤ poetry/pdm/… without uv on machine   → poetry run robotmcp  (needs rf-mcp as a project dep)

⑥ rf_conflict (irreconcilable)         → ATTACH BRIDGE (only conflict-free path):
                                          command = rf-mcp's own shim,
                                          env = {ROBOTMCP_ATTACH_HOST, ROBOTMCP_ATTACH_PORT,
                                                 ROBOTMCP_ATTACH_TOKEN}; project runs its own RF
                                          process (McpAttach) → all project libs available there
```

Detection precedence (most-specific marker wins): poetry > pdm > hatch > rye > pipenv > uv/PEP621
> plain-`.venv` > conda(active) > `VIRTUAL_ENV`(active) > bare-global. The authoritative signal is
the managing tool's `[tool.X]` table, not mere lockfile presence (hybrid dirs accrete stray locks).

## Resolved design decisions

1. **Overlay is the default; installing rf-mcp into the project env is opt-in (`--into-project`).**
   The `uv run --with` overlay is ephemeral and non-mutating (no lockfile/dep-tree change); it costs
   ~1s uv resolution per launch and needs network on first use. Installing rf-mcp *into* the project
   is faster and deterministic at launch but mutates the project's dependencies — so it is never
   silent.
2. **Hard version conflict → attach bridge, not a silent overlay.** Overlaying rf-mcp onto an RF<7
   project silently tests RF7. That is a correctness trap, so a detected `rf_conflict` routes to the
   attach bridge with a clear message (or `--into-project` is refused with the conflict explained),
   rather than writing a command that quietly tests the wrong versions.
3. **`--with rf-mcp==<own-version>` is pinned** to rf-mcp's installed version, killing the
   overlay's version-drift (a bare `--with rf-mcp` fetches latest).
4. **`--project <abs dir>` / `--python <abs env-py>` for cwd-independence.** Verified to work from
   an unrelated cwd; the config never relies on the agent setting cwd = project root. Absolute
   project paths are acceptable (portability is a non-goal; configs may be gitignored).
5. **`-m robotmcp.server` is the module entrypoint** for the INTO / interpreter forms (equivalent
   to the console script, no PATH dependency). A `robotmcp/__main__.py` MAY be added so `-m robotmcp`
   also works and routes through `entry` (nice-to-have, not required).

## Verification contract (never write a config that can't do the job)

Before persisting an entry, and unless `--no-verify`:
1. Spawn the resolved `command`+`args`+`env` with a clean, non-inherited environment.
2. Complete the MCP `initialize` handshake (must return `serverInfo`/capabilities).
3. Confirm at least one **detected** project library is reachable through rf-mcp
   (`check_library_availability` / `find_keywords` for that library reports it importable).
4. On failure: do **not** write the entry; report which step failed and the next action
   (e.g. "install rf-mcp into the env", "start the attach host", "reinstall rf-mcp[web]").

`robotmcp doctor --project [-C <dir>]` runs the same probe read-only and lists, per detected
project RF library, whether the currently-resolvable rf-mcp launch can see it. For attach-mode
entries, verification checks handshake + attach-host reachability (library reachability is deferred
to the running project process).

## Where today's installer falls short (all to be addressed)

1. `installer.py:resolved_command()` — only rf-mcp's own shim, no project awareness.
2. `installer.py:install()` — hardcodes `args=[]`, so it can't emit overlay/attach commands (though
   `adapters.build_entry(command, args, env)` already supports args/env).
3. `cli.py` — exposes no `--command`/`--project-dir`/`--env`/`--attach`/`--into-project`/`--no-verify`,
   so the existing `install()` kwargs are unreachable.
4. No project-env detector; `diagnostics.py` probes only rf-mcp's own interpreter.
5. `resolve_path` uses `Path.cwd()` unvalidated → project-scope config can land in the wrong dir.
6. No attach wiring, despite the bridge existing.

## Test strategy

- **Detector** unit tests over fixture project dirs (uv/plain-venv/poetry/pdm/pipenv/rye/hatch
  markers, conda via `CONDA_PREFIX`, bare-global) → correct type + interpreter + precedence.
- **Resolver** unit tests: each decision-tree branch produces the expected `command`/`args`/`env`
  (pinned `==<ver>`, `--project`/`--python`, attach env on conflict, own-shim on generic).
- **Verification** integration test: a project `.venv` with a project-only RF library — the resolved
  overlay command launches and reports that library reachable; a deliberately-broken command is
  rejected (nothing written). Reuse the MCP launch-probe pattern from exploration.
- **Conflict** test: RF<7 project → resolver selects attach (not overlay).
- `uv run pytest tests/unit` green; `openspec validate installer-project-aware-launch --strict`.
- Honest gap: pdm/pipenv/rye/hatch validated transitively (identical venv mechanism); a machine
  with those tools should close the loop.
