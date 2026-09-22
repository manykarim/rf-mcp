# Proposal: installer-cli-safety

## Why

Sandboxed probing of `robotmcp init | install | uninstall | list | doctor` with valid and
malformed arguments (evidence: `experiments/installer_ux_init_track.md`,
`experiments/installer_ux_install_track.md`, `experiments/adapter_config_audit.md`) found
defects in three classes: **actions taken that were promised not to be**, **failures
reported as success**, and **silent data loss in user config files**.

### 1. `--dry-run` is not side-effect-free (critical)

`robotmcp install --dry-run --yes --into-project -C <proj>` really runs
`uv pip install --python <proj>/.venv/bin/python rf-mcp==0.35.0` and mutates the project
environment. Proven twice — a shimmed `uv` captured the argv, then the real `uv` changed the
venv (`ModuleNotFoundError` before, import succeeds after). The output announces it while
still calling itself a dry run:

```
installed  (dry-run [into-project] installed rf-mcp into the project env; ...)   exit 0
```

Cause (verified in source): `install()` calls `resolve_launch(...)` at `installer.py:338`
**before any `dry_run` branch**, and `resolve_launch` calls `_install_into_project()` at
`installer.py:181`. `dry_run` is never passed down. Notably the verify step two lines below
*is* correctly gated (`if not (no_verify or dry_run)`), so dry-run was considered — just not
on the path that mutates.

### 2. Failure is reported as success, everywhere

`cli.py:86` assigns `rc = 0` once and returns it unchanged regardless of per-result status;
`cmd_init` and `cmd_doctor` end in unconditional `return 0`. Measured consequences:

- a refused install (`--command /nonexistent/bin/foo`) prints `unverified (...)`, writes no
  config, and exits **0**
- `robotmcp init` whose browser initialization **FAILED** exits **0**
- `robotmcp doctor` with zero test libraries and a nonexistent `-C` path exits **0**

Any scripted or CI use of these commands cannot detect a bad environment.

### 3. Malformed arguments are silently ignored

`adapters.py:159` does `a = BY_ID.get(token)` / `if a: out.append(a)` — unknown agent ids are
dropped without a word. Measured: `--agents claude` (the plausible spelling of `claude-code`)
→ `Nothing to do.`, exit 0, nothing written. So do `nosuchagent`, `ALL`, `Claude-Code`.
Worst form: `--agents claude-code,nosuchagent` installs one and never mentions the other.
Aggravated because `robotmcp list` prints display names (`Claude Code`) and never the ids the
flag actually takes. `--agents ""` silently means "detected" and installs into 5 agents, so
every malformed value is either nothing or everything — never an error. `--what MCP` reports
`no-assets (no bundled assets of this kind yet)`, indistinguishable from a legitimate result.
`--env FOO` and `--env =bar` are dropped silently. `--attach 99999` becomes hostname
`"99999"`. Only `--scope` gets this right (`invalid choice: 'global' ... ` exit 2) — it is
the model the rest should follow.

### 3b. Declining the confirmation prompt installs anyway (critical)

`robotmcp install` on a TTY asks `Register rf-mcp into these? [Y/n]`. Answering **n**
installs into every detected agent regardless. Reproduced on a real TTY: the user typed
`n` and all five adapters were written.

Cause: `_interactive_agents` (`cli.py:110`) returns `""` for a declined prompt, and
`adapters.resolve_selection` (`adapters.py:150`) opens with
`spec = (spec or "detected").strip()` — an empty spec is falsy, so "no" is silently
reinterpreted as "all detected agents". There is no code path by which the user's refusal
reaches a decision.

The same hole makes `--agents ""` mean "install everywhere", so a shell variable that
expands to nothing (`--agents "$AGENTS"`) silently installs into every detected agent.
Aborting with Ctrl-C is likewise unhandled and surfaces a `KeyboardInterrupt` traceback.

### 4. Config files are corrupted or written where the agent will not read them

- **Silent string corruption (JSONC).** `codecs.py:48` applies
  `re.sub(r",(\s*[}\]])", r"\1", ...)` to the whole document *including string literals*.
  Reproduced: `"Close the brace like this: { a, } and the bracket like [ b, ]"` is rewritten
  to `"... { a } ... [ b ]"`. The character scanner above it tracks string state correctly;
  the final regex discards that. Reachable for `fmt="jsonc"` (kilo) whenever the file has
  comments.
- **`kilo` writes a file Kilo does not read.** The adapter declares
  `project_path=".kilo/kilo.jsonc"` with `{"type":"stdio","command":"<string>"}`
  (`adapters.py:117`), while this repository's own working Kilo config is `.kilo/kilo.json`
  using `{"type":"local","command":["<argv>"]}`. Filename, `type`, and `command` cardinality
  all disagree; the install reports success.
- **Comment/anchor loss.** kilo JSONC destroys all comments (documented in `codecs.py`, but
  contradicted by `design.md:56`'s "preserving comments/formatting"). goose YAML destroys
  comments *and flattens anchors* — `defaults: &defaults` + `<<: *defaults` is expanded
  inline, so the DRY link is severed even though the data compares equal. That one is
  undocumented.
- **`uninstall -C dirA` removes dirB.** `uninstall()` filters only on
  `entries_for(wanted_ids, scope, whats)` (`installer.py:407`); the path is never a filter.
- **`uninstall` is gated on detection.** Bare `uninstall` defaults to `--agents detected`,
  so an entry installed into an undetected agent prints `Nothing to do.` and is left behind
  with a stale manifest row. The manifest already knows what was written.
- **Corrupt input crashes with a raw traceback** (`JSONDecodeError`, tomlkit `ParseError`),
  exit 1, path never named; on `uninstall` one corrupt file aborts the whole run.

### 5. The onboarding surface is undiscoverable, and one command hangs

- `robotmcp --help` prints the **MCP server's** parser: `init`, `install`, `uninstall`,
  `list`, `doctor` appear nowhere. Cause: `entry.py:14` routes on a `_SUBCOMMANDS` frozenset
  that omits help flags, so `--help` falls through to `server_main`.
- A typo'd subcommand (`instal`, `initt`, `help`) also falls through: wrong usage line, no
  suggestion, ~6x slower (2.4s vs 0.4s — it loads the whole server), preceded by unrelated
  library warnings that read like the cause of the failure.
- Bare `robotmcp` on a TTY prints one `ready` banner and then blocks forever on stdin. With
  no discoverable subcommands (above), that is the first thing a curious user sees.

### 6. `robotmcp init` downloads a browser without opt-in

`diagnostics.py:152` reads `want_browsers = browsers or libs.get("Browser")`. The `or` makes
the download branch fire whenever `robotframework-browser` is merely importable and
uninitialized — no flag, no prompt. Proven with a fake importable `Browser`: plain
`robotmcp init` printed `Initializing the Playwright browser (this downloads a browser...)`.
Both `--browsers`' own help text and `README.md:167` present that download as opt-in.

### 7. Diagnostics are incomplete and self-contradicting

- `diagnostics.py:13` `TEST_LIBRARIES` has no desktop/PlatynUI row, so `robotmcp doctor`
  cannot diagnose the product's most common install failure. On Python 3.10/3.11 the
  `python_version >= '3.12'` marker drops PlatynUI silently and nothing ever reveals it.
- `doctor -C <project>` can report `[ ] API (RequestsLibrary)   (add with rf-mcp[api])` and,
  twelve lines later, `extra project libraries: none (rf-mcp[all]'s bundle suffices)` for a
  project whose only RF dependency is `robotframework-requests`. `project_env._BUNDLED_DISTS`
  is a static catalogue of what `[all]` *would* provide, never intersected with what this
  install actually has.
- `robotmcp list`'s REGISTERED column reads rf-mcp's own manifest (`installer.py:442`), never
  the agent config — wrong in both directions. A user who pastes `init`'s snippet by hand is
  reported `no`; a user who deletes the config by hand is still reported `yes`.
- `doctor -C /nonexistent/xyz` and `-C /etc/passwd` print a clean report and exit 0; the
  path check in `cli.py` sits *after* the `doctor` branch has returned. `-C ""` silently
  inspects the CWD.

## What Changes

- **`--dry-run` becomes a hard guarantee**: thread `dry_run` into `resolve_launch`, and make
  every mutating helper refuse to run under it. Planned mutations are reported as planned.
- **Exit codes reflect outcomes**: non-zero when any result is a failure/refusal, when
  `init`'s browser step fails, and (for `doctor`) an opt-in `--strict`.
- **Declining or aborting is honoured**: answering `n` to the confirmation prompt, an
  empty `--agents` value, and Ctrl-C all stop the run with a clear message and a
  non-zero exit code, having written nothing.
- **Unknown argument values are errors, not silence**: unknown `--agents` ids, unknown
  `--what` kinds, malformed `--env` and `--attach` values are rejected with the offending
  token named and the valid set listed — matching `--scope`'s existing behaviour. `list`
  gains an ID column.
- **Config integrity**: fix the JSONC trailing-comma pass to skip string literals; align the
  `kilo` adapter with the format Kilo actually reads; preserve YAML anchors or refuse to
  rewrite anchored files; state comment-loss behaviour honestly per format.
- **`uninstall` correctness**: honour `-C`/scope when selecting entries, stop gating removal
  on detection, and continue past a corrupt file instead of aborting.
- **Readable failures**: corrupt config files produce a message naming the path and the
  parse error, not a traceback.
- **Discoverability**: route `-h/--help` (and unknown first tokens) to the onboarding parser
  with subcommands listed and a did-you-mean suggestion; make bare `robotmcp` on a TTY
  explain itself instead of silently blocking.
- **`init --browsers` becomes genuinely opt-in**; without it, report the browser as
  uninitialized and print the command to run.
- **Diagnostics**: add a desktop/PlatynUI row (with the correct install command from
  `install-extra-resolvability`), intersect the bundled-dist catalogue with what is actually
  installed, read the agent config for REGISTERED, and validate `-C` in all subcommands.

## Non-Goals

- The PlatynUI dependency resolution defects — covered by `install-extra-resolvability`.
- Adding new agent adapters or changing which agents are supported.
- Redesigning the launch-strategy logic (`uv-overlay` / `attach` / `own-shim`) itself.
