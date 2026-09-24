# Design: launch-env-fidelity

## One theme

All four defects are the same failure: **rf-mcp reports the state of the environment it is
running in, not the state of the environment the tests will run in.** Those are different
environments in every strategy except `own-shim`, and nothing currently reconciles them.

```
        what doctor inspects                 what the tests actually get
        ────────────────────                 ──────────────────────────
  D1    rf-mcp's own env: Browser OK    vs   overlay: no Browser
  D3    project pins RF 7.2             vs   overlay's RF 7.5 executes
  D4    return_code == 0 -> "passed"    vs   two libraries failed to import
  D2                                         a command uv refuses to run
```

## D1 - deriving the extras

The overlay spec must name the extras the running installation has. Three options:

| option | verdict |
|---|---|
| Hard-code `[all]` | **No.** A user who installed `rf-mcp[api]` would get the whole bundle layered onto their project - a much larger, slower overlay than they asked for, and a silent dependency change. |
| Read the extras from installed metadata | **Not possible.** Extras are not recorded in the installed distribution; `importlib.metadata` exposes `Provides-Extra` (what the package OFFERS) but not which were selected at install time. |
| **Probe which bundled libraries are importable, map back to extras** | **Chosen.** `project_env.BUNDLED_LIBRARIES` already maps library modules to what the extras provide, and `diagnostics.TEST_LIBRARIES` already carries the module -> extra mapping. Probing `find_spec` in rf-mcp's own interpreter is exactly what `doctor` already does. |

The third option is also self-correcting: it reflects what is genuinely importable, which is
the property that matters, rather than what was once requested on a command line.

Edge case: it can over-state. If a library is importable for a reason other than an rf-mcp
extra (a user `pip install`ed `robotframework-browser` into the tool env by hand), the
overlay will request the corresponding extra anyway. That resolves to the same package, so
the outcome is correct even though the inference was.

Edge case that must not regress: when NO bundled library is importable (a bare `rf-mcp`
install), the spec must stay extras-free rather than emitting `rf-mcp[]`.

## D2 - the directory check

One condition. `_rfmcp_with_args` already computes the path; it must additionally require
`Path(path).is_dir()` before choosing `--with-editable`, falling back to `--with <path>`.
A wheel path passed to `--with` is valid and resolves correctly - verified during
measurement (`--with <wheel>` and `--with <wheel>[all]` both resolved and ran).

Note the extras syntax differs by reference kind: `--with "<wheel>[all]"` works, and so does
`--with "rf-mcp[all]==<ver>"`. The editable form takes no extras suffix, which is a further
reason to reserve it for genuine source checkouts.

## D3 - how far to take the version check

`rf_conflict` deliberately guards only the major version, because that is where the
overlay is genuinely *unsafe* (RF 6 keywords against RF 7). A minor difference is usually
harmless, so turning it into a refusal would block working setups.

The fix is therefore **reporting, not refusal**: state which RF version will execute. This
keeps `rf_conflict`'s routing behaviour (major -> attach bridge) untouched and adds an
informational line for everything else. Anything stronger would be a behaviour change we
have no evidence justifies.

Determining the overlay's RF version requires resolving it, which is expensive. Prefer the
cheap comparison - the project's pin, parsed from its own metadata, against the RF version
rf-mcp itself has - and state the comparison's basis rather than pretending to a precision
we do not have.

## D4 - the status gate

```python
if return_code == 0:
    validation_status = "passed"      # wins unconditionally
elif issues:
    validation_status = "failed"
```

becomes error-first:

```python
if any(error-severity issues):
    validation_status = "failed"
elif return_code != 0:
    validation_status = "failed"
else:
    validation_status = "warning" if warnings else "passed"
```

`success` must follow `validation_status`; today they are computed independently, which is
how `success:true` and two error issues coexisted.

Risk: suites that currently report `passed` with benign parsed "errors" would start
reporting `failed`. The issue list is built by pattern-matching RF output
(`suite_execution_service.py:637`), so a false-positive match becomes a false failure. The
task list therefore includes auditing which patterns produce `severity: "error"` before
flipping the gate - the fix must not make validation noisy.

## Why this is separate from `installer-cli-safety`

That change was about the CLI's *own* behaviour - arguments, exit codes, what it writes.
This one is about whether the written launch *describes the environment it creates*. They
touch adjacent code but different questions, they were found by different methods (argument
probing vs. driving a real MCP client against a project fixture), and D4 lives in the
execution service rather than the onboarding package at all.

## Evidence

`experiments/tool_install_env_coupling_evidence.md` - measured with a real MCP stdio client
against a project fixture, cross-checked against plain `robot` runs as ground truth
(untracked; `experiments/` is git-excluded via `.git/info/exclude:17`).
