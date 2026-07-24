# rf-mcp 0.34.0

Three things drive this release: rf-mcp can now drive **native desktop apps** and be trusted on
**Windows**; the installer wires the server into **your project's own environment** so it sees your
libraries and keywords; and the guidance it feeds your agent is a lean checklist now, so runs waste
fewer turns. Plus a round of fixes to the things that wasted your time.

---

## Highlights

- **Native desktop automation is real and installable in one step** — `rf-mcp[desktop]`, PlatynUI
  under the hood, now Windows-ready (no more wedged keyboard, no minute-long stalls).
- **The installer understands your project** — install into a uv/poetry/venv project and rf-mcp runs
  against *that* project's libraries, keywords and resources, not its own.
- **Your agent wastes fewer turns** — the guidance rf-mcp hands the agent is a short checklist now,
  and it stops re-initializing sessions and fumbling API responses.
- **The first keyword no longer hangs** — a cold server used to stall ~90 seconds on step one. Now
  it answers in 0.03 s.

---

## Native desktop automation (PlatynUI)

rf-mcp can drive real desktop applications — Windows (UI Automation) and Linux (AT-SPI) — through
`PlatynUI.BareMetal` — keywords for pointer, keyboard, window management and XPath-style locators.
Two things in 0.34.0 make it practical:

**One-step install.** Desktop automation is now a proper extra, matched versions and all:

```
uv pip install "rf-mcp[desktop]"     # Python 3.12+; also included in rf-mcp[all]
```

Before, the PlatynUI packages were declared nowhere, so a routine `uv sync` would silently uninstall
your desktop stack — yesterday's working session would fail to load the library today. Now it's declared and version-matched.

**It works on Windows now.** A dedicated Windows 11 evaluation turned up four blockers; all fixed:

- **No more stuck keys.** A killed or timed-out run could leave a modifier key physically held down
  at the OS level — the operator's keyboard was wedged until reboot. rf-mcp now tracks exactly which
  keys it's holding and releases them on session end, on failure, and even after a force-kill.
- **Wrong locators fail in ~1.5 s, not 30–60.** A typo used to wait out PlatynUI's long default and
  stack across retries into multi-minute stalls. (Applies on Linux too.)
- **A slow query no longer freezes the server** — a `Focus`/`Query` on a large tree used to block
  everything, including unrelated calls.
- **Windows is allowed by default.** The safety guard only understood Linux's isolated-display model,
  so on Windows it refused *every* keyword, demanding an isolated display that doesn't exist there. Now it allows the active
  desktop with a one-time warning; set `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED=1` to opt back into refusal.

`rf-mcp[all]` now includes `desktop` on Python 3.12+, so the standard install gets it too.

---

## The installer understands your project

`robotmcp install` used to write a launch command pointing at rf-mcp's *own* environment. Because
rf-mcp imports Robot Framework libraries in its own process, that left the running server **blind to
your project's libraries** — a project `.venv` with, say, `JSONLibrary` came back as
`ModuleNotFoundError`, and your custom keywords and resources were invisible. It also never launched
the command it wrote, so a broken config was reported as "installed".

Installing globally is still effortless — `uvx` / `uv tool install` serves every project with no
per-project setup. What's new is that when you install **into a project that has its own
environment**, rf-mcp is wired to use *that* environment:

- **uv-first.** For a uv/poetry/pdm/pipenv/rye/hatch/plain-venv project, the installer points the server at your project's interpreter — without touching your project — so it sees your libraries:
  `uv run --no-project --python <your-venv>/bin/python --with rf-mcp robotmcp`.
- **Verified before it's written.** rf-mcp launches the resolved command and confirms your project's
  libraries are actually reachable *before* saving the config — a blind or broken command is refused,
  not persisted (`--no-verify` to skip).
- **Hard version conflicts** (a project pinning Robot Framework < 7) route to the **attach bridge**
  instead of testing a different RF version than the project pins.

New flags: `-C/--project-dir`, `--into-project` (opt in to installing rf-mcp into the project env),
`--attach`, `--command`, `--env`. And `robotmcp doctor --project-dir <dir>` reports which of your project's
libraries the resolved launch can see.

---

## Sharper agent steering

The guidance and tool descriptions rf-mcp gives your coding agent were reworked — this affects every
run, on every model.

- **Leaner default instructions.** The workflow guide rf-mcp hands the agent on connect is a short,
  order-explicit checklist now, instead of a ~2800-character essay that contradicted itself about
  which tool to call first. It also says the thing that needed saying: **`analyze_scenario` already
  creates the session**, so the agent stops re-initializing a session that already exists. Measured:
  redundant init calls dropped to zero, and a small model went from 1 of 3 runs passing to 3 of 3.
- **API sessions get the rules up front.** A session using RequestsLibrary now receives a compact
  cheat-sheet on how to read HTTP responses — agents used to miss this and fall back to dozens of
  `Evaluate` calls just to read a body. Roughly 35% fewer turns with it on. (`ROBOTMCP_API_GUIDANCE=off`.)
- **Rewritten tool descriptions.** Several MCP tool docstrings now lead with when to reach for the
  tool and its modes, so the agent picks the right one without trial and error.
- **"apple" is not a mobile app.** Scenario detection matched substrings, so `app` inside *apple*
  (and `ios` inside *kiosk*) turned an ordinary scenario into a mobile/Appium session. It matches
  whole words now.

Prefer the old text? `ROBOTMCP_INSTRUCTIONS_TEMPLATE=standard`.

---

## Fixes that unblock you

- **Cold-start hang (critical).** The first keyword against a freshly started server hung ~90–120 s
  (Robot Framework's start-up banner was corrupting the first MCP reply). Fixed: first step went from
  ~90 s to **0.03 s**, and a small-model run that used to time out now finishes in 28 seconds.
- **Windows dry-run deadlock (critical).** `run_test_suite` in dry-run mode froze for the full 180 s
  and timed out — every time, even for a one-line suite (an inherited-stdin deadlock, not your suite).
  Now ~1 second, Linux/macOS unaffected.
- **Windows `robotmcp init` / `install` crash.** On the Windows console (cp1252), non-ASCII output
  raised a `UnicodeEncodeError` and both commands failed hard. The onboarding output is ASCII-safe now.
- **Generated suites keep your paths.** Robot Framework treats `\` as an escape, so a Windows path
  written into a generated `.robot` was corrupted (`C:\WINDOWS\system32` → `C:WINDOWSsystem32`).
  Drive-letter paths now use forward slashes and other backslashes are escaped, so a suite that ran
  step-by-step also runs on replay.
- **Tool profiles work again.** On FastMCP 3 the small-context tool profiles had become a no-op — the
  server reported the profile "activated" while the model still saw the full tool surface. They now
  genuinely shrink the exposed tool set.

---

## Documentation

rf-mcp finally has real documentation beyond the README:

- **`docs/GETTING_STARTED.md`** — install, wire into an agent, run a first test.
- **`docs/MCP_TOOLS.md`** — the MCP tools, with parameters, returns and when to use each.
- **`docs/CONFIGURATION.md`** — the `ROBOTMCP_*` environment variables and CLI flags.
- **`docs/EXAMPLES.md`** — worked walkthroughs for Browser, Selenium, Requests, Appium, PlatynUI,
  BDD and data-driven suites.

---

## Defaults that moved (and how to restore them)

| Change | Restore the old behaviour |
|---|---|
| Default agent instructions are the short `lean` checklist | `ROBOTMCP_INSTRUCTIONS_TEMPLATE=standard` |
| Desktop query timeout is ~1.5 s (was ~30–60 s), all platforms | `ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS`, or `timeout_ms` per step |
| On Windows, desktop keywords are allowed by default | `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED=1` |
| RequestsLibrary sessions get an `api_guidance` block | `ROBOTMCP_API_GUIDANCE=off` |
| `rf-mcp[all]` now includes `desktop` on Python 3.12+ | — |
| Robot Framework console banners no longer emitted by the server | — |
| Generated suites write `C:/...` and escape backslashes | — (expect it in diffs of generated text) |

No MCP tool names, parameters or return shapes changed.

---

## Getting it

```
uv tool install "rf-mcp[all]"
robotmcp init
robotmcp install
```

`[all]` includes desktop automation on Python 3.12+ (no `--pre` needed). `pip install "rf-mcp[all]"`
works too. Upgrading from 0.33.x needs no configuration changes.
