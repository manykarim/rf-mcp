## 1. Primary fix — dry-run subprocess stdin

- [ ] 1.1 In `suite_execution_service.py` `run_robot_dry` (the `subprocess.run([python, -m, robot, --dryrun ...], capture_output=True, timeout=...)` call ~L329), pass `stdin=subprocess.DEVNULL` so the child does not inherit the server's pending-read stdin pipe.
- [ ] 1.2 Verify on Windows (or against the deterministic repro in `experiments/dryrun_windows_timeout_results.md`) that the dry-run returns in ~1 s instead of the 180 s timeout; verify no Linux regression (dry-run stays ~0.5–1 s).

## 2. Reap the whole process tree on timeout

- [ ] 2.1 On dry-run `subprocess.TimeoutExpired`, kill the child AND its descendants (Windows: `taskkill /F /T /PID`; POSIX: spawn with `start_new_session=True` and `os.killpg`) — a plain `kill()` orphans the uv-trampoline → real-interpreter chain and leaves the capture pipes open so `communicate()` hangs.
- [ ] 2.2 Make the reaper best-effort (never raise; fall back to `proc.kill()` if `taskkill`/`killpg` is unavailable). Confirm the timeout path returns a clean "timed out" error without itself hanging.

## 3. Close the latent second site + consistency

- [ ] 3.1 `utils/library_checker.py::check_pip_package_installed` — add `stdin=subprocess.DEVNULL` to the `pip list` subprocess (same inherited-stdin defect, 10 s fuse, runs during library checks).
- [ ] 3.2 `components/execution/platynui_focus.py` X11 probe subprocesses (~L1053/L1144) — add `stdin=subprocess.DEVNULL` for a single consistent policy (Linux-only/`DISPLAY`-gated, harmless there).
- [ ] 3.3 (Optional, per design open question) add a shared `run_subprocess_isolated(...)` helper that always sets a non-inheriting stdin, and route the above sites through it.

## 4. Defense-in-depth

- [ ] 4.1 Add `--console none` to the dry-run RF options in `_execute_rf_dry_run` (mirrors the in-process normal-run path) so RF console writes cannot reach an inherited/std handle.

## 5. Regression tests

- [ ] 5.1 Unit test: the dry-run subprocess is invoked with an explicit non-inheriting stdin (assert `stdin` is `DEVNULL`/not inherited) — mock `subprocess.run` and check the kwargs. Same for `library_checker`'s `pip list`.
- [ ] 5.2 Unit test: on a simulated `TimeoutExpired`, the timeout handler invokes the process-tree reaper and returns a structured timeout error without hanging.
- [ ] 5.3 Full unit suite green.

## 6. Docs + validation

- [ ] 6.1 Keep the evidence docs (`experiments/dryrun_windows_timeout_diagnostics.md`, `experiments/dryrun_windows_timeout_results.md`) as the acceptance reference; note the fix + before/after (180 s → ~1 s) where the dry-run subprocess is documented.
- [ ] 6.2 `openspec validate fix-mcp-subprocess-stdin-deadlock --strict` passes.
