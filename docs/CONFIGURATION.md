# Configuration Reference

rf-mcp runs with sensible defaults out of the box — install it, wire it into your
agent, and go. But defaults only get you so far. When you need to point it at a
running Robot Framework process, tighten the token budget for a small-context
model, or keep a desktop run from clicking on your real screen, everything is a
switch away.

This page is the full list of those switches: every `ROBOTMCP_*` environment
variable, the `robotmcp` command-line flags, and the onboarding subcommands. No
hidden knobs, no invented defaults — if it's here, it's in the source.

Environment variables are read from the process environment, so set them wherever
your MCP client launches the server (the `env` block of your MCP config, a shell
export, a `.env` your launcher loads). Values are case-insensitive unless noted,
and an unrecognised value falls back to the default rather than crashing the
server.

---

## Logging & output

Where the server's own logs go, and whether they reach the MCP client. Note that
stderr defaults to `WARNING` on purpose — MCP stdio transport lives on stdout, and
chatty logging there would corrupt the JSON-RPC stream.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, … | `WARNING` | Log level for the server's stderr logging. Case-insensitive. |
| `ROBOTMCP_MCP_LOG_NOTIFICATIONS` | any non-empty value | *(unset)* | When set, log records are also forwarded to the client as structured MCP log notifications (a middleware is attached at startup). |

---

## Instructions

The server ships MCP "instructions" — the workflow guide your agent reads on
connect. You can swap the template, disable them entirely, or point at your own
file. Handy when a model needs less hand-holding, or a lot more.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_INSTRUCTIONS` | `off`, `default`, `custom` | `default` | `off` disables instructions; `default` uses the selected template; `custom` loads `ROBOTMCP_INSTRUCTIONS_FILE`. Invalid values fall back to `default`. |
| `ROBOTMCP_INSTRUCTIONS_TEMPLATE` | `lean` (alias `checklist`), `minimal`, `standard`, `detailed`, `browser-focused`, `api-focused`, `desktop-focused`, `discovery_first`, `locator_prevention` | `lean` | Which built-in instruction template to serve in `default` mode. `lean` is a short, order-explicit checklist; set `standard` for the longer pre-0.34 text. Invalid values fall back to the default. |
| `ROBOTMCP_INSTRUCTIONS_FILE` | path (≤256 chars) | *(unset)* | Path to a custom instructions file. Required when `ROBOTMCP_INSTRUCTIONS=custom`; validated for path safety and existence. If unset in custom mode, falls back to `default`. |
| `ROBOTMCP_INSTRUCTION_MODE` | free-form string | `default` | Tag passed to the adaptive instruction-learning hooks at session start (labels the mode being observed). Distinct from `ROBOTMCP_INSTRUCTIONS`. |
| `ROBOTMCP_API_GUIDANCE` | `on`, `off` | `on` | Attach a compact RequestsLibrary cheat-sheet (`api_guidance`) to the session-start response when the session uses RequestsLibrary. Mirrors the existing desktop guidance. Set `off` to suppress it. |

---

## Attach bridge

rf-mcp can drive a *separate*, already-running Robot Framework process instead of
its own in-process context — the "attach bridge". Set `ROBOTMCP_ATTACH_HOST` and
the attach-aware tools forward to that process; leave it unset and everything runs
locally. Everything below only matters when you're attaching.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_ATTACH_HOST` | hostname / IP | *(unset)* | Host of the external RF bridge. **Setting this enables attach mode.** Unset ⇒ local execution. |
| `ROBOTMCP_ATTACH_PORT` | integer | `7317` | Port of the external RF bridge. |
| `ROBOTMCP_ATTACH_TOKEN` | string | `change-me` | Shared secret for the bridge handshake. Change it for anything but a throwaway local run. |
| `ROBOTMCP_ATTACH_DEFAULT` | `auto`, `off` | `auto` | `off` ignores the bridge even when a host is configured (forces local). `auto` uses the bridge when reachable. |
| `ROBOTMCP_ATTACH_STRICT` | `0`/`1`, `true`, `yes` | `0` | Strict mode: when the bridge is configured but a call fails, raise instead of silently falling back to local execution. |
| `ROBOTMCP_STARTUP_CLEANUP` | `auto`, `always`, `off` | `auto` | Startup session cleanup. `auto` cleans local sessions only when the bridge is healthy and its context is active; `always` cleans unconditionally; `off` disables it. Invalid values fall back to `auto`. |
| `ROBOTMCP_BRIDGE_HEARTBEAT` | `0`/`1`, `true`, `yes` | `0` | Enable the background bridge heartbeat/liveness probe. |
| `ROBOTMCP_HEARTBEAT_INTERVAL` | integer seconds | `60` | Seconds between heartbeat probes (when the heartbeat is enabled). |
| `ROBOTMCP_HEARTBEAT_THRESHOLD` | integer | `3` | Consecutive missed heartbeats before the bridge is treated as down. |

---

## Frontend dashboard

An optional Django-based web dashboard can run alongside the MCP server. It's off
by default; enable it with the environment flag or the `--with-frontend` CLI flag.
CLI flags win over environment variables when both are given.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_ENABLE_FRONTEND` | `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off` | `false` | Start the Django frontend alongside the MCP server. |
| `ROBOTMCP_FRONTEND_HOST` | hostname / IP | `127.0.0.1` | Interface the frontend binds to. |
| `ROBOTMCP_FRONTEND_PORT` | integer | `8001` | Frontend port. |
| `ROBOTMCP_FRONTEND_BASE_PATH` | path prefix | `/` | URL base path for the dashboard (a leading and trailing slash are enforced). |
| `ROBOTMCP_FRONTEND_DEBUG` | `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off` | `true` | Django debug mode for the frontend. |
| `ROBOTMCP_FRONTEND_DB` | path or `:memory:` | `:memory:` | SQLite database path for the frontend. In-memory by default (nothing persisted). |
| `ROBOTMCP_FRONTEND_STATIC_ROOT` | directory | *(bundled `static/` dir)* | Django `STATIC_ROOT` for collected static files. |
| `ROBOTMCP_FRONTEND_TIME_ZONE` | tz name | `UTC` | Django `TIME_ZONE`. |
| `ROBOTMCP_FRONTEND_SECRET_KEY` | string | *(random per start)* | Django secret key. A random key is generated each start if unset — set it explicitly for anything persistent. |
| `ROBOTMCP_FRONTEND_EVENT_BUFFER` | integer | `2048` | Size of the in-memory event buffer feeding the live dashboard. |

> The frontend needs its extra dependencies: `pip install "rf-mcp[frontend]"`.

---

## Memory

Persistent semantic memory (ADR-014) lets the server remember useful patterns
across sessions. It's **off by default** and, when on, runs on a small torch-free
embedding model. Requires `pip install "rf-mcp[memory]"`.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_MEMORY_ENABLED` | `true`/`1`/`yes`, else off | `false` | Master switch for persistent semantic memory. |
| `ROBOTMCP_MEMORY_DB_PATH` | path | `~/.rf-mcp/memory.db` | SQLite (sqlite-vec) database file for stored memories. |
| `ROBOTMCP_MEMORY_MODEL` | `potion-base-8M`, `all-MiniLM-L6-v2` | `potion-base-8M` | Embedding model. Dimension is derived (256 for `potion-base-8M`, 384 for `all-MiniLM-L6-v2`; unknown models default to 256). |
| `ROBOTMCP_MEMORY_MAX_RECORDS` | integer | `10000` | Max records kept per collection before pruning. |
| `ROBOTMCP_MEMORY_PRUNE_DAYS` | number (days) | `90` | Age past which records become prune candidates. |
| `ROBOTMCP_MEMORY_DECAY_HALF_LIFE` | number (days) | `30` | Time-decay half-life applied when ranking recalled memories. |
| `ROBOTMCP_PROJECT_ID` | string | `default` | Namespaces stored memories per project, so unrelated projects don't cross-recall. |

---

## Output & token economy

rf-mcp is deliberately frugal with tokens — large tool outputs can be spilled to
disk and referenced, schemas can be slimmed, and the whole tool surface can be
tuned to a model's context window. This is where you dial that in.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_OUTPUT_MODE` | `inline`, `file`, `auto` | `auto` | How large tool outputs are handled. `inline` always returns full content; `file` externalizes to an artifact; `auto` externalizes only when the content exceeds the inline threshold. Invalid values fall back to `auto`. |
| `ROBOTMCP_OUTPUT_VERBOSITY` | `compact`, `standard`, `verbose` | `standard` | Verbosity of formatted responses. `compact` abbreviates field names and trims detail. |
| `ROBOTMCP_MAX_INLINE_TOKENS` | integer | `500` | Approx. token threshold above which output is externalized (in `auto`/`file` modes). |
| `ROBOTMCP_FETCH_ARTIFACT` | `true`/`1`/`yes`, else off | `false` | Enable the `fetch_artifact` tool so the agent can pull externalized content back. Only takes effect when output mode is not `inline`. |
| `ROBOTMCP_TOKENIZER` | `heuristic`, `cl100k_base`, `o200k_base` | `heuristic` | Token-estimation backend. The `tiktoken` backends need `pip install "rf-mcp[tokens]"`; without it, falls back to the heuristic (chars ÷ 4). |
| `ROBOTMCP_CATALOG_HARD_CAP` | integer | `100` | Hard cap on the number of keywords returned in a discovery catalog response. |
| `ROBOTMCP_TOOL_PROFILE` | `browser_exec`, `api_exec`, `desktop_exec`, `discovery`, `minimal_exec`, `slim_exec`, `full` | *(auto-selected)* | Default tool profile — which tools are exposed — when `manage_session` doesn't specify one. On FastMCP 3.x a profile changes the tool *set* only; per-profile description and input-schema trimming is not applied. |
| `ROBOTMCP_MODEL_TIER` | `small_7b`, `small_context`, `medium_13b`, `standard`, `large_context`, `hosted` | *(auto)* | Model-capability tier hint used to auto-select a profile when `manage_session` doesn't pass one. |

---

## Library detection & keyword ranking

How rf-mcp guesses which Robot Framework library a scenario needs, and how it ranks
`find_keywords` results. The defaults are tuned; reach for these only when
detection is fighting you.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_LIBRARY_DETECTION_MIN_SCORE` | integer | `5` | Minimum score for a library to be considered a detection candidate. |
| `ROBOTMCP_LIBRARY_DETECTION_CONFLICT_THRESHOLD` | integer | `8` | Score gap above which one library clearly wins (no conflict). |
| `ROBOTMCP_LIBRARY_DETECTION_AMBIGUITY_WINDOW` | integer | `4` | Score window within which candidates are treated as ambiguous (ties). |
| `ROBOTMCP_SEMANTIC_KEYWORDS` | `1`/`true`/`yes`, else off | *(off)* | Enable semantic (embedding-based) keyword ranking in `find_keywords`. Off ⇒ lexical ranking. Lazy-loads a torch-free `model2vec` backend on first use. |
| `ROBOTMCP_MATCHER_RERANK` | `1`/`true`/`yes` vs `0`/`false`/`no` | `1` (on) | Enable the keyword-matcher re-ranking pass. |
| `ROBOTMCP_RERANK_CAP` | float | `0.5` | Ceiling applied to re-rank score adjustments. |
| `ROBOTMCP_RERANK_DOWNWEIGHT` | float | `0.6` | Down-weight factor applied to de-prioritized keyword matches. |

---

## PlatynUI desktop safety

Desktop automation is powerful and, unmanaged, dangerous — it drives the real
pointer and keyboard. These variables relax or tighten the guards, and rf-mcp
only clicks inside a scoped target.

The default depends on the platform:

- **Linux** — rf-mcp refuses to act on an active/unknown display and only
  proceeds on a provably isolated one. **Loosen this only when you know the run
  is isolated** (e.g. Xvfb/Xephyr).
- **Windows** — there is no nested-display isolation model, so a Windows host is
  classified as `windows` and allowed by default, with a one-time warning that
  it is driving the live desktop. Use `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED` to
  refuse instead.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_PLATYNUI_SAFETY_GUARD` | `warn` | *(enforce)* | `warn` downgrades the desktop safety guard from refuse-to-run to log-a-warning. Any other value keeps enforcement. |
| `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP` | `1`/`true`/`yes`, else off | *(off)* | Allow input on the active/primary desktop instead of refusing. Off ⇒ only a proven-isolated display is allowed. Not needed on Windows, which is allowed by default. |
| `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED` | `1`/`true`/`yes`, else off | *(off)* | Strict opt-in for Windows: refuse desktop interaction keywords on the active Windows desktop instead of allowing them. No effect on Linux, which already refuses non-isolated displays. |
| `ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS` | integer milliseconds (>0) | `1500` | Default timeout for desktop element queries/waits, on **all platforms**. Replaces PlatynUI's own ~30s (60s for broad queries) default so a wrong locator fails fast instead of stacking retries. Raise it if you rely on a long implicit wait; an explicit `timeout_ms` on a step overrides it for that call. |
| `ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY` | display string (e.g. `:99`) | *(unset)* | Marker recorded by the bootstrap: the isolated `DISPLAY` it provisioned. Corroborates that input is safe. |
| `ROBOTMCP_PLATYNUI_ISOLATED_XPID` | PID | *(unset)* | PID of the X server (Xvfb/Xephyr/Xorg) that owns the isolated display. The guard verifies it's a live X server for that display, so a stale marker can't false-allow input. |
| `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED` | `1`/`true`/`yes`, else off | *(off)* | Permit a deliberate desktop-wide (unscoped) element search; otherwise scoping to the app under test is enforced (one-time warning on opt-in). |
| `ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW` | `1`/`true`/`yes`, else off | *(off)* | Opt out of the Linux `control:Window` fail-fast guard (which prevents a 30s hang). |
| `ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR` | `1`/`true`/`yes`, else off | *(off)* | Opt out of the guard that rejects a bare file path handed where an element descriptor is expected (e.g. `Take Screenshot`). |
| `ROBOTMCP_PLATYNUI_NO_FOCUS` | `1`/`true`/`yes`, else off | *(off, focus on)* | Disable focus-before-act (the default raises/focuses the target window before interacting). |
| `ROBOTMCP_PLATYNUI_HIGHLIGHT` | `0`/`false`/`no` disables | *(on)* | Element-highlight overlay during interaction. Set to a falsy value to turn it off. |
| `ROBOTMCP_PLATYNUI_KEEP_WAYLAND` | `1`/`true`/`yes`, else off | *(off, shim active)* | Keep the Wayland session as-is. By default rf-mcp forces `XDG_SESSION_TYPE=x11` to avoid a Wayland portal hang. |
| `ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT` | float seconds (>0) | `5.0` | Cap on PlatynUI descriptor-resolution time during batch retries, so a bad descriptor can't stall the batch. Bad/≤0 values fall back to the default. |
| `ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE` | `warn` | *(enforce)* | `warn` downgrades the steering-confidence gate to a warning instead of enforcing it. |

---

## Artifacts & screenshots

Where externalized tool output and screenshots land on disk, and whether a
screenshot may be returned to the agent as an image block.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_ARTIFACT_DIR` | directory | `.robotmcp_artifacts` | Directory where externalized (large) tool outputs are written. |
| `ROBOTMCP_ARTIFACT_TTL` | integer seconds | `3600` | Retention time-to-live for artifacts before cleanup. |
| `ROBOTMCP_SCREENSHOT_DIR` | directory | *(system temp dir)* | Directory for captured screenshots. |
| `ROBOTMCP_SCREENSHOT_MODE` | `file`, `image`, `auto` | `file` | Governs `visual_check`: `file` returns a path only; `image`/`auto` may return a FastMCP image content block (when the tool is asked for one). Invalid values fall back to `file`. |

---

## Misc

Startup timing, plugin discovery, execution routing, and the odds and ends that
don't fit a group above.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `ROBOTMCP_LAZY_INIT` | `0` disables | *(lazy on)* | Defer building the execution engine until first use (faster handshake). Set to `0` to build it eagerly at import. |
| `ROBOTMCP_WARMUP` | `0` disables | *(warm-up on)* | Background warm-up thread that materializes the engine after startup so the first tool call doesn't pay the init cost. Ignored when `ROBOTMCP_LAZY_INIT=0`. |
| `ROBOTMCP_PLUGIN_PATHS` | `os.pathsep`-separated paths | *(bundled manifest dir)* | Extra directories to scan for library-plugin manifests. |
| `ROBOTMCP_RF_RUNNER_REQUESTS` | `1`/`true` vs `0` | `1` (on) | Route RequestsLibrary session operations through the RF runner. Set `0` to disable. |
| `ROBOTMCP_RF_CONTEXT_ONLY` | `1`/`true` vs `0` | `1` (on) | Execute keywords through the RF context only. Set `0` to allow non-context execution paths. |
| `ROBOTMCP_PRE_VALIDATION` | `1`/`true` vs `0` | `1` (on) | Pre-validate keyword arguments before execution. Set `0` to disable. |
| `ROBOTMCP_USE_SAMPLING` | `true`/`1`/`yes`, else off | `false` | Enable MCP sampling (ask the client's LLM for help on select tasks). |
| `ROBOTMCP_LLM_TYPE` | free-form string | *(auto-detected)* | Override the detected client model identifier (used for adaptive tuning). Falls back to `ANTHROPIC_MODEL` / `OPENAI_MODEL`, then `unknown`. |
| `ROBOTMCP_DISABLE_LEARNING` | `1`/`true`/`yes`, else on | *(learning on)* | Disable the adaptive instruction-learning hooks. |

---

## Command-line reference

`rf-mcp` installs the `robotmcp` command. Bare `robotmcp` launches the MCP server;
the onboarding subcommands (`init`, `install`, `uninstall`, `list`, `doctor`) and
`--version` are handled *before* the server starts and exit without launching it.

### Server flags

Run with no subcommand to start the server. These flags configure transport and
the optional frontend (CLI flags override the corresponding environment
variables).

```bash
# stdio (default) — what MCP clients launch
uv run -m robotmcp.server

# HTTP transport with the dashboard alongside
uv run -m robotmcp.server --transport http --host 127.0.0.1 --port 8000 --with-frontend
```

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--transport` | `stdio`, `http`, `sse` | `stdio` | MCP transport. |
| `--host` | hostname / IP | `127.0.0.1` | Host/interface for HTTP transport. |
| `--port` | integer | `8000` | Port for HTTP transport. |
| `--path` | path | `/` | Path for HTTP/streamable endpoints. |
| `--log-level` | e.g. `INFO`, `DEBUG` | *(see `ROBOTMCP_LOG_LEVEL`)* | Log level for the server. |
| `--with-frontend` | flag | off | Start the Django frontend alongside the server. |
| `--without-frontend` | flag | — | Disable the frontend even if the environment enables it. |
| `--frontend-host` | hostname / IP | `127.0.0.1` | Host for the frontend server. |
| `--frontend-port` | integer | `8001` | Port for the frontend server. |
| `--frontend-base-path` | path prefix | `/` | Base path prefix for the frontend. |
| `--frontend-debug` | flag | — | Enable Django debug mode for the frontend. |
| `--frontend-no-debug` | flag | — | Disable Django debug mode for the frontend. |

### Subcommands

These are for installing and health-checking rf-mcp itself — they never start the
server.

| Command | Effect |
|---------|--------|
| `robotmcp --version` (`-V`) | Print the installed rf-mcp version and exit. |
| `robotmcp init [--browsers]` | Idempotent, non-destructive setup check: reports which test libraries are present, optionally runs the Playwright browser download (`--browsers`), and prints the MCP config snippet to paste into your agent. |
| `robotmcp doctor` | Read-only health report: version, executable path, which test libraries are importable, whether the Playwright browser is initialized, and whether Node.js is on PATH. |
| `robotmcp list` | List supported coding agents and their status (detected / registered / config format). |
| `robotmcp install` | Register rf-mcp into detected (or specified) coding agents. |
| `robotmcp uninstall` | Remove rf-mcp's entries from agents. |

#### `install` / `uninstall` flags

| Flag | Values | Default | Effect |
|------|--------|---------|--------|
| `--agents` | `all` \| `detected` \| comma-separated ids | `detected` | Which agents to target. |
| `--scope` | `project`, `user` | `project` (install) | Write into project-local or user-global config. |
| `--what` | comma list of `mcp,skills,agents,hooks` | `mcp` (install) / `mcp,skills,agents,hooks` (uninstall) | Which artifacts to write/remove. |
| `--dry-run` | flag | off | Show the plan; write nothing. |
| `--yes` / `--no-input` | flag | off | Non-interactive; don't prompt. |
| `--force` | flag *(install only)* | off | Overwrite an existing `robotmcp` entry. |

---

## See also

- **Optional dependencies (pip extras):** `web`, `api`, `mobile`, `database`,
  `frontend`, `memory`, `tokens`, `semantic`, `all` — install with
  `pip install "rf-mcp[web,api]"` etc. See the README for the full matrix.
- **README** — install, MCP client wiring, and worked examples.
