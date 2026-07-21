## 1. Primary fix — dry-run subprocess stdin

- [x] 1.1 `suite_execution_service.py` `run_robot_dry` now spawns via `subprocess.Popen(..., stdin=subprocess.DEVNULL, ...)` so the child never inherits the server's pending-read stdin pipe.
- [x] 1.2 Verified on Linux: the file dry-run of the exact Windows-Calculator suite returns in ~0.6 s, `success=True` (no regression). The Windows deadlock repro (§8 of the results doc) confirmed `stdin=DEVNULL` fixes the hang.

## 2. Reap the whole process tree on timeout

- [x] 2.1 On `communicate(timeout=...)` `TimeoutExpired`, `run_robot_dry` calls `_kill_process_tree(proc)` (Windows `taskkill /F /T /PID`; POSIX `killpg` — the child is spawned with `start_new_session=True`) BEFORE draining, so an orphaned uv-trampoline grandchild can't hold the capture pipes open.
- [x] 2.2 `_kill_process_tree` is best-effort: swallows all errors and falls back to `proc.kill()`; the timeout path returns the structured "timed out" error without hanging. Covered by `test_kill_process_tree_never_raises` + `test_dry_run_timeout_reaps_process_tree`.

## 3. Close the latent second site + consistency

- [x] 3.1 `utils/library_checker.py::check_pip_package_installed` — `pip list` subprocess now passes `stdin=subprocess.DEVNULL`.
- [x] 3.2 `components/execution/platynui_focus.py` X11 probe subprocesses (L1055/L1146) — `stdin=subprocess.DEVNULL` added (Linux-only, consistent policy).
- [x] 3.3 Decided against a shared `run_subprocess_isolated` wrapper: the sites are heterogeneous (dry-run needs Popen + tree-kill; the others are simple `subprocess.run`). The reaping is factored into `_kill_process_tree`; the stdin isolation is a one-line kwarg per site, and the regression tests assert the call shape so a future omission fails.

## 4. Defense-in-depth

- [x] 4.1 Considered `--console none` for dry-run and DEFERRED it: unlike the normal-run path, the dry-run's stdout is a **captured pipe** that the validation parser reads (there is no `output.xml` — `--output NONE`), so `--console none` blanks `suite_info`/`test_count` and breaks validation. The `stdin=DEVNULL` isolation (§1) already fixes the deadlock; there is no inherited handle for the console to leak to here. Documented the reason inline in `_execute_rf_dry_run`.

## 5. Regression tests

- [x] 5.1 `tests/unit/test_subprocess_stdin_isolation.py`: the dry-run `Popen` and the `pip list` `subprocess.run` are both asserted to pass `stdin=subprocess.DEVNULL`. `test_dry_run_from_file_options.py` also asserts the dry-run `Popen` stdin.
- [x] 5.2 Simulated `TimeoutExpired` → the reaper is invoked and a structured timeout error is returned without hanging; the reaper never raises on a bogus pid.
- [x] 5.3 Full unit suite green: **7094 passed, 4 skipped** (updated `test_dry_run_from_file_options.py` + `test_suite_relative_resource_from_file.py`, which mocked the old `subprocess.run`, to mock `Popen`).

## 6. Docs + validation

- [x] 6.1 Evidence retained as the acceptance reference (`experiments/dryrun_windows_timeout_diagnostics.md`, `experiments/dryrun_windows_timeout_results.md`); the fix rationale is documented inline in `run_robot_dry` / `_kill_process_tree`.
- [x] 6.2 `openspec validate fix-mcp-subprocess-stdin-deadlock --strict` passes.
