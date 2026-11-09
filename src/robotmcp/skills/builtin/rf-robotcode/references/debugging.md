# Debugging a real test with rf_debug

Use `rf_debug` whenever an existing test is involved. Rebuilding the test in `rf_repl` runs a copy without the real suite setup, variables and imports.

## Start

```
rf_debug(action="start", by_longname=["Tests.Auth.Login Works"])
```

- By default the run pauses at the first uncaught failing keyword (before the stack unwinds).
- `breakpoints=["login.robot:42"]` - the line must hold a keyword call, not the test-name line. `breakpoints=["Submit Login"]` breaks on a keyword (globs and `Lib.Keyword` work).
- Always combine a breakpoint with the run scope (`by_longname`); a bare path stops at the first failure in the file, which may be another test.
- `stop_on_entry=true`, `no_break_on_exception=true`, `break_on_all_exceptions=true`, `break_on_failed_test=true`, `break_on_failed_suite=true` map to the `robot-debug` flags of the same names.
- Output files: `rf_debug` writes none by default (`--output NONE --log NONE --report NONE`), so the last run's `output.xml` stays usable for `rf_results` and `rerunfailed`. Pass `keep_outputs=true` only when you need the debug run's files.

A `status="error"` with `run_exit_code=252` means the run did not start - read `output`, fix the selection or options.

## At the stop

Choose each command from the previous answer; do not script a sequence blindly.

| Need | `command` |
|---|---|
| Where am I? | `.where` (`.w`), `.list`, `.source` |
| Move between frames | `.up`, `.down`, `.frame N` |
| Variables | `.vars --user`, `.print ${x}`, `.pprint ${x}`, `.whatis ${x}`, `.set`, `.display ${x}` (see `.help <cmd>`) |
| Step | `.step`, `.next`, `.return`, `.until`, `.continue` (`.c`) |
| Breakpoints | `.break <loc>[, <cond>]`, `.tbreak`, `.breakpoints`, `.condition`, `.ignore`, `.delete`, `.disable`, `.enable`, `.commands` (logpoints), `.catch uncaught|all|test|suite|off` |
| End the stop | `.continue`, `.detach`, or `.abort` (exit code 253) |
| Try a fix | any keyword line, e.g. `Should Be Equal    ${actual}    expected` - it runs in the paused frame |

Debugger commands start with a dot; bare `c`, `n` or `where` would be run as keywords (rf-mcp adds the dot for the common ones).

A fix tried at the prompt is gone when the run ends. Write it into the `.robot` / `.resource` file, `rf_analyze`, and rerun by longname.

## Sessions

- `action="status"` says whether the run is running, waiting at the prompt, or ended; `action="result"` gives its exit code.
- A long step returns `status="running"` after `timeout_seconds`; continue with `action="wait"` or stop it with `action="interrupt"`.
- `action="restart"` reruns with the same breakpoints and picks up edited Python libraries and resource files.
- A debug session stays open after its run ends; idle sessions end after 15 minutes. Stop it with `action="stop"`, or tell the user it is still open.
- Client mode: if the agent shell kills background processes, the next call says `No session '<id>' is open`. Run the one-shot `fallback_command` once and report only what its output shows - never try to escape or reconfigure the sandbox, and never claim values you did not see.
