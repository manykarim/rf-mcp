# rf-mcp 0.34.0

Since 0.33.0 we took rf-mcp to Windows and watched it struggle. This release is what we fixed.

---

## Highlights

- **The first keyword works again.** A cold server used to stall ~90 seconds on your very first
  step. Now it answers in 0.03.
- **Windows dry runs finish.** `run_test_suite` in dry-run mode timed out at 180 s on Windows —
  every time, even for a one-line suite. Now ~1 second.
- **Desktop automation stopped stealing your keyboard.** A killed run could leave a key physically
  held down at the OS level. That is now tracked and released — every key, even after a force-kill.
- **Tool profiles actually shrink the tool set.** On FastMCP 3.x they had become a no-op, so
  small-context models kept paying for the full tool surface.

---

## Desktop automation

Native GUI automation with PlatynUI. **Platform scope:** the safety-guard change is Windows-only;
the query-timeout and stuck-key changes apply on **Linux too**.

### No more stuck keys

Desktop keys are injected at the OS input level. If a key goes down and never comes up, it stays
down — system-wide, after your run is over, after the process is gone. On Windows this happened: a
modifier stayed held, the operator could no longer press Enter or the Windows key, and the fix was a
reboot. Not our finest hour.

rf-mcp now tracks exactly which keys it is holding and releases that exact set when a session ends,
a keyboard keyword fails, or the server exits.

- **Every key, not just modifiers.** A bare `Keyboard Press A` or `<F12>` sends key-down only —
  those are released now too. The old Ctrl/Alt/Shift/Win sweep remains as a fallback. (Space and Tab
  are covered as `<Space>` / `<Tab>` tokens.)
- **Survives a force-kill.** `Stop-Process -Force` and `SIGKILL` skip every cleanup handler. Held
  keys are recorded to a small per-process file, and the next desktop session start lifts anything a
  dead rf-mcp process left behind. No operator intervention, no reboot.
- **Your deliberate chords are safe.** A `Keyboard Press` held on purpose for a later
  `Keyboard Release` in the same live session is never released early, and a healthy concurrent
  session's keys are never touched by another process.
- **Steering, not nagging.** A failed desktop keyboard step now tells you held keys were released
  and points at the atomic `Keyboard Type    <Ctrl+A>` instead of a bare `Keyboard Press`.

### Wrong locators fail in seconds, not minutes

*(All platforms.)* A typo used to cost you real time: desktop queries fell back to PlatynUI's own
~30 s default (60 s for broad queries), and retries stacked to roughly three minutes. The default is
now ~1.5 s.

- `ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS` — raise or lower the new short default.
- `timeout_ms` on `execute_step` is honoured for desktop keywords, applies to that call, and no
  longer sticks to the rest of the session.

If you were relying on the long implicit wait — waiting for an app window to appear, say — pass an
explicit `timeout_ms` on that step. **Linux desktop users upgrading from 0.33.x should audit steps
that depended on the old 30 s wait.**

### A slow query no longer freezes the server

A `Focus` or `Query` on a large UI tree used to block everything. Even a pure metadata call like
`get_keyword_info` would hang behind it for up to a minute. Native desktop queries now run in the
background, so the rest of the server keeps answering while one is in flight.

### Windows is a supported desktop, not a rejected one

The active-desktop safety guard only understood X11. On Windows that meant every interaction keyword
was refused, and the suggested remedy was an Xephyr/Xvfb recipe, which does not exist on Windows.

Windows hosts are now classified as such and allowed by default, with a one-time warning that you
are driving the live desktop and an isolation note that actually applies (dedicated user session or
RDP).

- `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP` is no longer needed on Windows. Unchanged on Linux.
- `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED` — new strict opt-in if you want refusal back.

### Installing it is one extra now

A routine `uv sync` used to uninstall your desktop stack — yesterday's working session would start
failing to load the library, because none of it was declared anywhere. You also had to keep the
Robot Framework library and its Rust core at matching versions by hand.

The new `desktop` extra declares all of it: PlatynUI, its native Rust core and the `platynui-cli`
diagnostic tool, as version-matched prebuilt wheels. See [Getting it](#getting-it).

- Requires Python 3.12+. On 3.10/3.11 the extra resolves and installs nothing.
- **`rf-mcp[all]` now includes `desktop`** on Python 3.12+, so the standard install
  gets desktop automation without a second step. (`semantic` remains excluded — it
  pulls torch.)
- PlatynUI's keyword surface grows to ~30 keywords.

---

## Windows, generally

### Dry-run validation no longer hangs

`run_test_suite` in dry-run mode froze for the full 180 s timeout and returned
`Dry run execution timed out after 180s`. Every time. Even for `Log    hi`.

It looked like a desktop-suite problem. It wasn't — the validation subprocess was sharing the MCP
server's own input channel, which deadlocks on Windows regardless of what your suite contains.
Dry-run validation now returns in about a second, with no change on Linux or macOS.

The same class of hang is fixed in two quieter places: the installed-package check behind
`check_library_availability` / `recommend_libraries`, and the Linux display probes. And when a dry
run genuinely does time out, rf-mcp now shuts down the whole process tree it started instead of
leaving an orphaned `robot`/`python` process behind.

### Generated suites keep your paths

Robot Framework treats `\` as an escape character, so a Windows path written into a generated suite
was destroyed the moment the file was parsed. `C:\WINDOWS\system32` came back as
`C:WINDOWSsystem32`. `C:\Users\name\report.txt` got a real newline and carriage return baked into
the middle of it.

The result was a scenario that ran perfectly step by step and then failed on replay with a confusing
"cannot find the file" — while the saved `.robot` looked fine at a glance.

- **Drive-letter paths** are now written with forward slashes: `C:/WINDOWS/system32/calc.exe`. Runs
  as recorded on Windows and Linux.
- **Everything else with a backslash** is now escaped so it survives parsing. `Should Match Regexp`
  with `\d+` used to reach the keyword as `d+`; `data\output.txt` used to arrive as `dataoutput.txt`.
  Regenerating an already-escaped suite produces identical text.

Only generated files were ever affected — live step execution passes arguments straight to the
keyword and never went through the parser.

**Known limitation.** A relative path whose separator is immediately followed by `n`, `r` or `t`
(`data\new`) is genuinely ambiguous with Robot Framework's real `\n`/`\r`/`\t` and is not recovered.
Use forward slashes or an absolute path for those.

---

## Starting up

### The cold-start hang is gone

The very first keyword against a freshly started server used to hang for roughly 90–120 seconds.
Every later step was instant, which made it look like a fluke rather than a bug.

Robot Framework was writing its test-start banner directly onto the same channel the server uses to
answer your MCP client. That garbled the first reply, and the client sat there until it gave up. In
agent-driven runs this burned the entire time budget, or surfaced as a misleading "tool use was
rejected" — nothing was ever rejected, the call had simply been killed after timing out.

Measured after the fix: first step on a cold server went from ~90 s to **0.03 s**, and a small-model
end-to-end run that used to time out now finishes in 28 seconds with a suite that runs and passes.

Robot Framework console banners from the server process are switched off entirely — if you were
watching for them, they are gone. Keyword results, logs and generated suites are untouched.

---

## Steering your agent

### Leaner default instructions

The instructions rf-mcp hands your coding agent on connect are now a short checklist that says which
tool to call first, instead of a ~2800-character workflow essay that contradicted itself on exactly
that point.

**`analyze_scenario` already creates the session** — the agent must not also call
`manage_session(action="init")`. Agents were routinely re-initialising a session that already
existed, because both tools advertised themselves as "the start".

Measured: redundant session-init calls dropped to zero, and a small model went from completing 1 of
3 runs to 3 of 3.

- `ROBOTMCP_INSTRUCTIONS_TEMPLATE` — default moved from `standard` to `lean`. Every previous
  template (`minimal`, `standard`, `detailed`, `browser-focused`, `api-focused`, `desktop-focused`,
  `discovery_first`, `locator_prevention`) is still selectable.

### API sessions get the cheat-sheet automatically

Sessions using RequestsLibrary now receive a compact API guidance block in the session-start
response — the non-obvious response-access rules, delivered before the agent needs them. Agents used
to miss these entirely and fall back to dozens of `Evaluate` calls just to read a response body.

Measured with the guidance on versus off: 58 turns against 88.7 on average, roughly 35% fewer.

- `ROBOTMCP_API_GUIDANCE` — `on` (default) / `off`.

### "apple" is not a mobile app

Automatic platform detection matched substrings, so the token `app` matched inside *apple* and
*application*, and `ios` matched inside *kiosk*. A scenario like "create a list of three fruits
(apple, banana, cherry) and verify its length is 3" was turned into a mobile session with
AppiumLibrary loaded.

Detection now matches whole words. Genuine mobile scenarios still detect as mobile. Note that only
`context="desktop"` overrides the classifier outright; other `context` values are hints.

---

## Tool profiles

Tool profiles exist to shrink the tool surface for small-context models. On FastMCP 3.x they had
become a no-op: you could set a profile, the server would report it as activated, and the tool list
the model actually saw never changed. No error, no warning — just a context budget going nowhere.

Profiles now genuinely reduce the exposed tool set, and switching profiles repeatedly inside one
server process is stable (it used to be able to break every subsequent tool call).

- `ROBOTMCP_TOOL_PROFILE`, `ROBOTMCP_MODEL_TIER`, and the `tool_profile` / `model_tier` /
  `model_name` parameters on `manage_session(action="init")`.
- **Scope note:** profiles change *which* tools are exposed, not their descriptions or input
  schemas. Per-profile description and schema trimming is not done on FastMCP 3.x. Fewer tools reach
  the model, which is the token saving the feature was for.

---

## Documentation

rf-mcp finally has documentation beyond the README.

- **`docs/GETTING_STARTED.md`** — install, wire into a coding agent, run a first test.
- **`docs/MCP_TOOLS.md`** — 51 MCP tools in 9 groups, with parameters, returns and when to use them.
  Only a subset is exposed by default; the rest are legacy or gated behind extras and profiles.
  Previously the only way to see a tool's parameters was to get an agent to call it and read what
  came back.
- **`docs/CONFIGURATION.md`** — the `ROBOTMCP_*` environment variables with defaults, plus the
  `robotmcp` CLI commands and flags. (The three variables introduced in this release —
  `ROBOTMCP_API_GUIDANCE`, `ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS`, `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED`
  — are documented here in these notes and are still being folded into that reference.)
- **`docs/EXAMPLES.md`** — seven worked walkthroughs: Browser, SeleniumLibrary, RequestsLibrary,
  Appium, PlatynUI desktop, BDD-style and data-driven.

The README now leads with the three-command setup:

```
uv tool install "rf-mcp[all]"
robotmcp init
robotmcp install
```

Hand-written MCP JSON still works and is kept in a "legacy / manual config" section.

---

## Behaviour changes worth knowing

Most of this release is fixes, but a few defaults moved. In rough order of "might surprise you":

| Change | Restore the old behaviour |
|---|---|
| Desktop query timeout is now ~1.5 s, **all platforms** (was PlatynUI's ~30 s / ~60 s) | `ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS`, or pass `timeout_ms` per step |
| On Windows, desktop keywords are allowed by default (previously refused) | `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED=1` |
| Default agent instructions are the short `lean` checklist | `ROBOTMCP_INSTRUCTIONS_TEMPLATE=standard` |
| RequestsLibrary sessions get an extra `api_guidance` block | `ROBOTMCP_API_GUIDANCE=off` |
| Generated suites write `C:/...` and escape backslashes | — (this is the fix; expect it in diffs of generated text) |
| Robot Framework console banners no longer emitted by the server | — |
| Scenarios containing "apple"/"application"/"kiosk" no longer become mobile sessions | Use explicit mobile wording, or `manage_session(action="init", libraries=["AppiumLibrary"])` |
| Teardown releases all tracked held keys (all platforms), and a session start clears keys left by a dead process | — |
| Tool profiles change tool *count*, not descriptions or schemas | — |

No MCP tool names, parameters or return shapes changed. Some tool descriptions were rewritten
(shorter, with the load-bearing guidance first).

---

## Under the hood

The end-to-end suite now drives rf-mcp with autonomous agents and fails the build when tool
descriptions or workflow guidance regress — so a future release cannot make rf-mcp harder for your
agent to drive.

---

## Getting it

0.34.0 is still baking. The work above ships today as `0.34.0.devN` prereleases:

```
uv tool install --prerelease=allow "rf-mcp[all]"
```

`[all]` now includes desktop automation on Python 3.12+ — no second command. On Python 3.10/3.11 it
installs everything else and skips desktop.

Want desktop on its own, without the rest:

```
uv tool install --prerelease=allow "rf-mcp[desktop]"
```

Upgrading from 0.33.x needs no configuration changes. If a moved default bites you, the table above
has the switch.
