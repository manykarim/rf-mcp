## Why

`robotmcp install` writes a launch command that points at rf-mcp's **own** environment
(`resolved_command()` → `shutil.which("robotmcp")`). Because rf-mcp imports Robot Framework
libraries via `namespace.import_library()` in its **own process `sys.path`**, the running server
is then **blind to the libraries installed in the user's project** (proven: a project `.venv`
with `robotframework-jsonlibrary` → `ModuleNotFoundError: JSONLibrary` under rf-mcp's own Python).
An agent asked to drive that project's custom keywords fails.

The installer also never launches the command it writes, so a config that doesn't actually start
the server — or starts one that can't see the project's libraries — is persisted as "installed"
with no signal. That is unacceptable for a tool whose whole job is to drive the project's tests.

## What Changes

- **Project-env detection.** Add a detector that, for `--scope project` (and a new
  `-C/--project-dir`), identifies the project's Python environment (uv/plain-venv/poetry/pdm/
  pipenv/rye/hatch/conda/global) and its interpreter.
- **uv-first launch resolver.** Choose a `command`+`args` so the running rf-mcp sees the project's
  RF libraries, preferring uv:
  - rf-mcp already importable in the project env → run it there (`uv run --project <dir> robotmcp`
    or the project's own shim). Best case: no overlay, no mutation.
  - project is a virtualenv (any tool) + uv present + extra libs + no conflict → **uv overlay**
    (`uv run --project <dir> --with rf-mcp==<ver> robotmcp`, or `--python <env-py>` for
    non-uv venvs). Non-mutating, cwd-independent. **The default.**
  - no project env / only libraries `rf-mcp[all]` bundles → rf-mcp's own shim (today's behaviour).
  - **Fallbacks** (uv can't overlay a non-venv prefix, or no uv): co-install into a conda/global
    env, or `poetry run`/`pdm run`.
  - **Hard version conflict** (project pins RF<7 / fastmcp<3 / python<3.10) → wire the **attach
    bridge** (rf-mcp isolated, drives the project's own RF process) — the only conflict-free path.
- **Pre-write launch verification.** Before persisting, launch the resolved command, complete the
  MCP `initialize` handshake, and confirm a detected project library is reachable through rf-mcp;
  only write if it passes (`--no-verify` to skip). Add `robotmcp doctor --project` to report which
  of the project's RF libraries rf-mcp can currently see.
- **New CLI flags + wiring.** `-C/--project-dir`, `--into-project` (opt-in mutation), `--attach`,
  `--command`, `--env`, `--no-verify` — routed into the existing `install()` kwargs, and
  `install()` stops hardcoding `args=[]`.

## Capabilities

### Modified Capabilities
- `tool-install-onboarding`: the installer becomes project-aware — it resolves a launch command
  (uv-first) that lets the running server see the target project's Robot Framework libraries,
  verifies the command launches and reaches those libraries before writing, and falls back to the
  attach bridge under an irreconcilable dependency conflict.

## Impact

- **Code:** `onboarding/installer.py` (project-aware resolver + non-empty args + verification hook),
  new `onboarding/project_env.py` (detector), `onboarding/cli.py` (new flags), `onboarding/adapters.py`
  (unchanged entry shapes — already accept args/env), `onboarding/diagnostics.py` (`doctor --project`).
- **Behaviour:** project-scope installs now produce a server that can drive the project's own RF
  libraries; a broken or blind command is caught before it is written. User-scope / no-project
  installs are unchanged (STEP: own shim). Nothing is mutated in the project env unless
  `--into-project` is passed.
- **Non-goals:** portability of the written config (the user accepts machine-specific, gitignore-able
  configs); auto-installing rf-mcp into the project env by default (opt-in only).
