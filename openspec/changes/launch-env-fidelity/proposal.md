# Proposal: launch-env-fidelity

## Why

rf-mcp imports Robot Framework libraries into its **own** process and runs suites
**in-process** (`project_env.py:4`; `library_manager.py:315` `importlib.import_module`;
`suite_execution_service.py` builds a `robot.api.TestSuite`). So the rf-mcp interpreter IS
the test-execution interpreter, and the launch command written into an agent's config
decides which libraries and which Robot Framework version the user's tests actually get.

Measuring that coupling against a realistic project (evidence:
`experiments/tool_install_env_coupling_evidence.md`) found four defects where **what
rf-mcp reports does not match the environment the tests will run in**.

The project fixture: a `.venv` pinning `robotframework>=7.0`, a non-bundled RF library
(`robotframework-jsonlibrary`), a custom `libs/AcmeLibrary.py`, and suites importing it
both module-based and path-based.

### D1 - the uv overlay drops the extras the user installed (high)

`_rfmcp_with_args()` (`installer.py:82`) returns `["--with", f"rf-mcp=={ver}"]` - **no
extras**. So a user who runs the README's headline command and then registers a project:

```
uv tool install "rf-mcp[all]"      # Browser, Selenium, Appium, Requests, Database
robotmcp install -C <project>      # resolves to the uv-overlay strategy
```

gets a launch that layers **bare** rf-mcp onto the project env. Measured end-to-end through
a real MCP client:

```
overlay as rf-mcp writes it   available:['JSONLibrary']
                              missing  :['Browser','SeleniumLibrary']
overlay with extras           available:['Browser','SeleniumLibrary','JSONLibrary']
                              missing  :[]
```

The user installed `[all]`, `robotmcp doctor` reports Browser and SeleniumLibrary as
present (they are - in the tool env), and the configured launch silently has neither.
Web automation cannot run, and nothing says why.

### D2 - `--with-editable <wheel>` is not a valid uv argument (medium)

`_rfmcp_with_args` returns `--with-editable <path>` whenever `direct_url.json` carries a
`file:` URL, without checking whether that path is a DIRECTORY. An install from a local
wheel therefore produces a launch command uv rejects outright:

```
error: Editable must refer to a local directory, not an archive:
  `file:///.../rf_mcp-0.35.1-py3-none-any.whl`
```

The install-time verification gate does catch this (it refuses to write an unverified
command), so the user is blocked rather than silently broken - but with a uv error, not an
explanation. This is the install shape used by the new CI installability guard.

### D3 - the overlay's Robot Framework shadows the project's (medium)

In an overlay launch the RF that executes the tests comes from the overlay, not the project:

```
robot from      : /home/many/.cache/uv/archive-v0/.../site-packages/robot/__init__.py
JSONLibrary from: .../proj/.venv/lib/python3.12/site-packages/JSONLibrary/__init__.py
```

`rf_conflict` (`project_env.py`) guards only the MAJOR version (`RF_MIN_MAJOR = 7`), so a
project pinned to `robotframework==7.2` is silently tested on the overlay's 7.5. The user
is told the overlay lets rf-mcp "see" their libraries; they are not told it also replaces
their RF.

### D4 - dry-run reports "passed" while imports fail (medium)

`suite_execution_service.py:644`:

```python
if return_code == 0:
    validation_status = "passed"
elif issues:
    validation_status = "failed"
```

`return_code == 0` wins unconditionally, so error-severity issues are discarded. Measured
on a tool-installed rf-mcp validating the fixture suite:

```
success:true   validation_status:"passed"   return_code:0
imports_valid:false
issues: [error] "Failed to import library 'AcmeLibrary'"
        [error] "Failed to import library 'JSONLibrary'"
```

An agent reading `validation_status:"passed"` proceeds to a full run and only then
discovers the environment is wrong - exactly the signal that should have stopped it. This
is the same class as the exit-code defect fixed in `installer-cli-safety`: an outcome that
means "failed" reported as "succeeded".

## What Changes

- **The overlay carries the extras rf-mcp actually has (D1).** Derive the extras from the
  running installation (which of the bundled libraries are importable) and include them in
  the `--with` spec, so the overlay environment provides the same libraries the user
  installed. Where they cannot be derived, say so rather than silently dropping them.
- **`--with-editable` is used only for a real directory (D2).** A `file:` URL pointing at
  an archive resolves to a normal `--with <path>` instead, which is valid and correct for a
  wheel.
- **The overlay's Robot Framework version is reported, and a mismatch is surfaced (D3).**
  Extend the conflict check beyond the major version: when the project pins an RF version
  the overlay would not satisfy, say which version will actually execute the tests, in
  `doctor` and in the install-time note.
- **Validation status reflects error-severity issues (D4).** `validation_status` is
  `failed` whenever an error issue is present, regardless of `return_code`; `success`
  follows it. Warnings keep their current behaviour.
- **Document the environment coupling (all four).** The README explains, briefly, that
  rf-mcp runs the tests in its own interpreter, what that means for an existing RF project,
  and that hand-pasting the `init` snippet bypasses the project-aware launch that
  `robotmcp install` resolves.

## Non-Goals

- **D5 - `--into-project` changing the project's pinned RF version.** Measured
  (`uv pip install rf-mcp` into an RF 6.1.1 project resolves
  `- robotframework==6.1.1 / + robotframework==7.5`), but out of scope here at the user's
  direction. It is a real hazard on an explicit, opt-in, documented-as-mutating flag, and
  should get its own change.
- Changing the launch-strategy decision tree itself (`uv-overlay` / `in-project` /
  `attach` / `own-shim`). The strategies are sound; only their fidelity is at issue.
- The module-based `libs/` import gap. Verified to be **identical to plain `robot`**
  behaviour (fails without `PYTHONPATH` in both, passes with it in both), so it is a
  project-configuration matter, not an rf-mcp defect.
