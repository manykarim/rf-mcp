# Tasks — installer-project-aware-launch

## 1. Project-environment detector
- [x] 1.1 Add `onboarding/project_env.py`: given a project dir, detect env type by
      most-specific marker (poetry > pdm > hatch > rye > pipenv > uv/PEP621 > plain-`.venv`
      > conda(active) > `VIRTUAL_ENV`(active) > bare-global) and return `{type, python_path, is_venv}`.
- [x] 1.2 Resolve each type's interpreter (`.venv/bin/python`, `poetry env info --path`,
      `pdm info --python`/`.pdm-python`, `pipenv --venv`, `hatch env find`, `CONDA_PREFIX`,
      system python). Best-effort; never raise — return `None`/`bare-global` on failure.
- [x] 1.3 Detect `extra_libs`: RF libraries the project references that `rf-mcp[all]` does NOT
      bundle — from the project env's installed packages AND `Library`/`Resource` imports scanned
      in the project's `.robot`/`.resource` files (local `.py` libs aren't pip-visible).
- [x] 1.4 Detect `rf_conflict`: project pins Robot Framework < rf-mcp's floor, an incompatible
      shared dep, or a Python older than rf-mcp's baseline.

## 2. uv-first launch resolver
- [x] 2.1 In `installer.py`, add `resolve_project_command(project_dir, *, into_project, attach)`
      implementing the decision tree: ① rf-mcp-in-project → project shim / `uv run --project`;
      ② venv + uv + extra_libs + !conflict → `uv run --project <dir>|--python <env-py> --with
      rf-mcp==<own-ver> robotmcp`; ③ generic/no-env → `resolved_command()` (own shim);
      ④ conda/global fallback → co-install or `conda run`; ⑤ no-uv fallback → `poetry/pdm run`;
      ⑥ conflict → own shim + attach env. Returns `(command, args, env)`.
- [x] 2.2 Pin the overlay to rf-mcp's installed version (`--with rf-mcp==<importlib.metadata
      version>`); use `--project <abs dir>` / `--python <abs env-py>` for cwd-independence.
- [x] 2.3 Stop hardcoding `args=[]` in `install()`; thread the resolved `(command, args, env)`
      into `ad.build_entry(command, args, env)`. Keep user-scope / `--command` overrides working.

## 3. Verification before writing
- [x] 3.1 Add a launch probe: spawn `(command, args, env)` with a clean env, complete the MCP
      `initialize` handshake, and confirm a detected project library is reachable via rf-mcp
      (`check_library_availability` / `find_keywords`). Return pass/fail + the failing step.
- [x] 3.2 Gate the write on verification (unless `--no-verify`): on failure, do not write; report
      the failed step and the remedy. Record "verification skipped" in the manifest when skipped.
      For attach entries, verify handshake + attach-host reachability only.
- [x] 3.3 Add `robotmcp doctor --project [-C <dir>]`: read-only report of which project RF
      libraries the currently resolvable launch can see.

## 4. CLI flags + wiring
- [x] 4.1 Add to `cli.py` install/uninstall parsers: `-C/--project-dir`, `--into-project`,
      `--attach`, `--command`, `--env` (repeatable `KEY=VALUE`), `--no-verify`; route into the
      existing `install()` kwargs (`command=`, `env=`, `cwd=`, plus new ones).
- [x] 4.2 Validate `--project-dir` looks like a project (pyproject/.git/.venv); warn instead of
      silently writing to the wrong directory. Default project dir = cwd (as today) but validated.
- [x] 4.3 `--into-project` performs the opt-in install into the detected env
      (`uv add`/`uv pip install`/`pip`), then resolves to the in-project shim (STEP ①).

## 5. Tests & validation
- [x] 5.1 Detector unit tests over fixture project dirs (each env type + precedence + conda/global).
- [x] 5.2 Resolver unit tests: each decision-tree branch → expected `command`/`args`/`env`
      (pinned `==<ver>`, `--project`/`--python`, attach env on conflict, own-shim on generic).
- [x] 5.3 Verification integration test: project `.venv` with a project-only RF library → resolved
      overlay launches and reports it reachable; a deliberately-broken command is refused (no write).
- [x] 5.4 Conflict test: RF<7 project → resolver selects attach, not overlay.
- [x] 5.5 `uv run pytest tests/unit` green; `openspec validate installer-project-aware-launch --strict`.
