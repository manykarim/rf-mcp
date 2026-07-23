# Getting Started with rf-mcp

rf-mcp is an MCP server that hands your AI coding agent a live Robot Framework
session. The agent discovers keywords, executes steps one at a time against real
libraries — Browser, Selenium, Appium, Requests, Database, PlatynUI — inspects
the DOM and page state as it goes, and builds a runnable `.robot` suite from what
actually worked. No guessing, no hallucinated locators.

One thing to keep straight: rf-mcp is *not* a Robot Framework library of keywords.
Its "keywords" are its MCP tools. It's the bridge, not the toolbox.

This guide takes you from zero to a passing test. Grab a coffee.

## Prerequisites

| You need | Why |
| --- | --- |
| **Python 3.10+** | rf-mcp and Robot Framework 7.0+ run here. |
| **Node.js** | Only for the Browser (Playwright) library. Skip it if you're doing API, Selenium, or database work. |
| **A coding agent** | Claude Code, Codex, Copilot, Cursor, Gemini CLI, and friends — see [Register into coding agents](#3-register-into-coding-agents). |

rf-mcp ships lean. The core install pulls minimal dependencies; you add exactly
the test libraries you want through extras.

## 1. Install as a tool

The cleanest path is to install rf-mcp as a standalone CLI tool. `uv` drops it in
its own isolated environment and puts a `robotmcp` command on your PATH.

```bash
# Everything: Browser, Selenium, Appium, Requests, Database, ...
uv tool install "rf-mcp[all]"

# Or just what you need
uv tool install "rf-mcp[api]"        # API testing only, pure Python
uv tool install "rf-mcp[web,api]"    # Web + API
```

Extras control which Robot Framework libraries are available and what, if
anything, you have to do afterward:

| Extra | Adds | Post-install |
| --- | --- | --- |
| `api` | RequestsLibrary | none |
| `web` | SeleniumLibrary + Browser | Selenium: none (Selenium Manager fetches the driver); Browser: `robotmcp init --browsers` |
| `mobile` | AppiumLibrary | Appium server (external) |
| `database` | DatabaseLibrary | a DB driver |
| `frontend` | Django dashboard | — |
| `memory` | Persistent semantic memory (sqlite-vec + model2vec) | — |
| `desktop` | PlatynUI native desktop automation (Rust core + `platynui-cli`) | Python 3.12+; ships as a pinned prerelease |
| `all` | all of the above (Robot Framework libraries) | as above |

> **`desktop` needs Python 3.12+.** It is included in `[all]`, but on Python
> 3.10/3.11 it resolves to nothing — `[all]` still installs fine, you just don't
> get desktop automation there.

> `[all]` deliberately leaves out the `semantic` extra — that one pulls
> sentence-transformers and torch (~2GB) as an optional max-quality backend for
> keyword ranking. You almost certainly don't need it; the default ranker is
> torch-free and just as sharp on the metric that matters.

Prefer `pip`? `pip install "rf-mcp[all]"` works too. The tool-install route just
keeps rf-mcp's dependencies from tangling with your project's.

## 2. Prepare and diagnose

Two commands get you set up and tell you the truth about your environment.

```bash
robotmcp init            # reports libraries, prints the MCP config to paste
robotmcp init --browsers # also initializes the Playwright browser (needs Node.js)
robotmcp doctor          # read-only health check: version, libraries, browser, Node
robotmcp --version
```

`robotmcp init --browsers` runs the bundled `rfbrowser init` *inside* rf-mcp's own
environment, so Playwright lands exactly where the installed Browser library looks
for it. It advises rather than fails if the `web` extra or Node.js is missing — no
dead ends, just a clear next step.

`robotmcp doctor` never changes anything. Run it any time something feels off; it
reports version, detected libraries, browser status, and Node without touching a
byte.

## 3. Register into coding agents

rf-mcp can wire itself into your agent's config so you don't have to hand-edit
JSON.

```bash
robotmcp list                              # supported agents + what's detected/registered
robotmcp install                           # interactive: registers into detected agents
robotmcp install --agents claude-code,codex,gemini
robotmcp install --agents all --scope user
robotmcp install --dry-run                 # show the plan, write nothing
robotmcp uninstall                         # safe, reversible removal
```

**Supported agents:** Claude Code, OpenAI Codex, GitHub Copilot, opencode,
Gemini CLI, Kilo Code, goose, and Cursor. Each is written in its own file and
format, and any other MCP servers you already have configured are left untouched.

**Scope.** Installs default to `--scope project` (writes into the current project,
e.g. `./.mcp.json`) where the agent supports it. Use `--scope user` for a global,
home-directory install. goose only supports user scope.

**Safe and reversible.** Every change is recorded in a hash-tracked manifest at
`~/.local/state/robotmcp/install-manifest.json`. `robotmcp uninstall` removes only
the entries that are unchanged since install — edit an rf-mcp entry by hand and it
gets left in place and reported instead of clobbered. Unrelated servers are never
touched. Use `--dry-run` on either command to preview.

**Manual fallback.** Prefer to edit config yourself? Add this to your agent's MCP
config:

```json
{ "mcpServers": { "robotmcp": { "command": "robotmcp" } } }
```

For Claude Code specifically, one line does it:

```bash
claude mcp add rf-mcp -- uvx rf-mcp
```

## 4. Your first test

Restart your agent so it picks up the new MCP server, then just ask. rf-mcp guides
the agent through discover → execute → verify → build, so plain English is enough:

```
Use rf-mcp to build and run a test for https://www.saucedemo.com/ that:
- Logs in with a valid user
- Adds two items to the cart
- Completes checkout
- Verifies the success message

Use SeleniumLibrary. Execute the steps one at a time, then build the final suite.
```

Behind the scenes, the agent works through the rf-mcp tools roughly like this:

1. **`manage_session`** — opens a session and loads SeleniumLibrary + BuiltIn.
2. **`find_keywords`** / **`get_locator_guidance`** — finds the right keywords and
   real locator syntax before touching the page.
3. **`execute_step`** — runs each step live (open browser, fill login, click,
   assert), inspecting the DOM with **`get_session_state`** when it needs to find
   an actual id or label.
4. **`build_test_suite`** — turns the steps that passed into a clean `.robot`
   file.
5. **`run_test_suite`** — runs the finished suite end to end to prove it's green.

You watch it happen, step by step. When it's done you have a real, runnable Robot
Framework suite built from executions that actually worked — not a hopeful draft.

## Where to go next

- **[MCP_TOOLS.md](MCP_TOOLS.md)** — the full reference for all rf-mcp tools:
  signatures, parameters, and what each one does.
- **[CONFIGURATION.md](CONFIGURATION.md)** — environment variables, instruction
  templates, transports (STDIO / HTTP), and the frontend dashboard.
- **[EXAMPLES.md](EXAMPLES.md)** — worked scenarios across web, API, mobile,
  database, and desktop.

Built by the community, for the community. Now go automate something.
