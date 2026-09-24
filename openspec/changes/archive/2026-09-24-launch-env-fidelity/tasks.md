# Tasks: launch-env-fidelity

## 1. D1 - the overlay carries the installation's extras
- [x] 1.1 Add a helper that returns the extras the running rf-mcp provides, by probing which of `project_env.BUNDLED_LIBRARIES` are importable and mapping back through `diagnostics.TEST_LIBRARIES`' module -> extra mapping
- [x] 1.2 `_rfmcp_with_args` (`installer.py:63`) includes those extras in the `--with` spec; emit no bracket suffix when the set is empty (never `rf-mcp[]`)
- [x] 1.3 Verify the extras syntax for each reference kind actually resolves: `rf-mcp[web,api]==<ver>` (published) and `<path>.whl[web,api]` (local wheel)
- [x] 1.4 When the extras cannot be determined, append a caveat to `LaunchPlan.note` instead of presenting the overlay as complete
- [x] 1.5 Unit test: a fake env where Browser+Selenium are importable yields a spec containing `web`
- [x] 1.6 Unit test: a bare install (no bundled library importable) yields a spec with no bracket suffix
- [x] 1.7 End-to-end: overlay-launched rf-mcp reports Browser AND the project's non-bundled library as available (reproduces the measurement in the evidence file)

## 2. D2 - editable only for a directory
- [x] 2.1 `_rfmcp_with_args`: require `Path(path).is_dir()` before returning `--with-editable`; otherwise return `--with <path>`
- [x] 2.2 Unit test: a `file:` direct-URL pointing at a `.whl` yields `--with`, not `--with-editable`
- [x] 2.3 Unit test: a `file:` direct-URL pointing at a source directory still yields `--with-editable`
- [x] 2.4 Confirm the CI installability guard's local-wheel install now resolves a runnable project-aware launch

## 3. D3 - report the executing Robot Framework version
- [x] 3.1 Extend the project-pin parse to capture the full version specifier, not only the major version
- [x] 3.2 Where the resolved launch would not satisfy that pin, add a line naming the RF version that will execute the tests
- [x] 3.3 Surface it in `robotmcp doctor -C <project>` and in the install-time resolved-launch note
- [x] 3.4 Leave `rf_conflict`'s major-version routing to the attach bridge unchanged (it is the only case where the overlay is genuinely unsafe)
- [x] 3.5 Unit tests: pin satisfied -> silent; pin unsatisfied by minor version -> reported, not refused; major mismatch -> still routed to attach

## 4. D4 - validation status reflects error issues
- [x] 4.1 Audit which patterns in `suite_execution_service.py:637` produce `severity: "error"`; confirm none are benign, so the stricter gate cannot make validation noisy
- [x] 4.2 Replace the `return_code == 0` short-circuit (`suite_execution_service.py:644`) with an error-first gate
- [x] 4.3 Make `success` follow `validation_status` rather than being computed independently
- [x] 4.4 Unit test: error-severity issues + `return_code == 0` -> failed / unsuccessful
- [x] 4.5 Unit test: warnings only -> not failed
- [x] 4.6 Unit test: clean validation -> passed / successful
- [x] 4.7 Regression check across the existing suite-validation tests for newly-failing expectations, and reconcile each deliberately

## 5. Documentation - the environment coupling
- [x] 5.1 README: short subsection explaining that rf-mcp imports RF libraries into its own process and executes suites there
- [x] 5.2 README/GETTING_STARTED: what that means for an existing RF project, and that `robotmcp install` run FROM the project directory resolves a project-aware launch
- [x] 5.3 Add the caveat next to the hand-paste MCP snippet: it uses rf-mcp's own environment and bypasses project-aware resolution
- [x] 5.4 `robotmcp init`'s printed snippet carries the same one-line caveat
- [x] 5.5 Release notes entry covering D1-D4, leading with D1 (installed extras silently absent from an overlay launch)

## 6. Verification
- [x] 6.1 Re-run the strategy matrix from `experiments/tool_install_env_coupling_evidence.md`; every row's "MISSING" that this change targets must flip
- [x] 6.2 Confirm the attach-bridge and RF-6 conflict paths are unchanged
- [x] 6.3 Full unit suite green
