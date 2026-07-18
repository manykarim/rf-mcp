## 1. Foundation — `robotmcp init` / `doctor` / `--version` / alias

- [x] 1.1 Add an `init` subcommand to `server.py`'s parser (dispatch before `mcp.run()`; `init` must NOT start the server).
- [x] 1.2 `init` detects installed test libraries via `importlib.util.find_spec`; when Browser is present (or `--browsers`) it runs `subprocess.run([sys.executable, "-m", "Browser.entry", "init"])` with a "downloading…" message and a venv-local `rfbrowser` fallback.
- [x] 1.3 `init` advises (prints `uv tool install "rf-mcp[web]"`) instead of failing when the web extra is absent; warns (not fails) when Node.js is missing; always prints the `{"command":"robotmcp"}` MCP snippet. Idempotent.
- [x] 1.4 `robotmcp --version` reads from `importlib.metadata.version`; read-only `robotmcp doctor` reports version, per-library import status, browser-init state, Node presence, resolved executable path.
- [x] 1.5 Add a thin `robotmcp.entry:main` (dispatches onboarding without importing the heavy server) and point both `robotmcp` and the new `rf-mcp` `[project.scripts]` at it.

## 2. Agent-adapter registry

- [x] 2.1 Define an `AgentAdapter` interface: `id`, `detect()`, `config_path(scope)`, format codec (JSON/JSONC/TOML/YAML), `render_server_entry(command, args, env)`, `insert()/remove()` on parsed config.
- [x] 2.2 Implement adapters: Claude Code, Codex, GitHub Copilot, opencode, Gemini CLI, Kilo Code, goose, Cursor (per the design's config-path/format table).
- [x] 2.3 Register `pi` with `status="planned"` (adapter body stubbed) until its MCP-config convention is confirmed; `list` surfaces status.
- [x] 2.4 Pull in comment/format-preserving codecs where needed (e.g. `tomlkit`, `ruamel.yaml`, a JSONC-aware reader) as optional install deps.

## 3. Installer CLI — `install` / `uninstall` / `list`

- [x] 3.1 `robotmcp list` shows registry adapters, detected-on-machine status, and per-agent current rf-mcp registration state.
- [x] 3.2 `robotmcp install` with `--agents all|detected|<csv>`, `--scope project|user`, `--what mcp[,skills,agents,hooks]`, `--dry-run`, `--yes/--no-input`, `--force`; interactive mode pre-checks detected agents.
- [x] 3.3 `install --what mcp` merges the `robotmcp` server (resolved absolute executable path + opt-in env) into each targeted agent's config without overwriting other servers; existing rf-mcp entry updated only with `--force`.
- [x] 3.4 `--what skills,agents,hooks` recognized as no-op selectors that report "no bundled assets yet" (extension seam), leaving the manifest schema and per-agent asset paths ready.
- [x] 3.5 `robotmcp uninstall` reverts only manifest entries whose current hash still matches; user-edited entries are left and reported; supports the same `--agents/--scope/--what/--dry-run` flags.

## 4. Manifest

- [x] 4.1 Manifest store (e.g. `~/.local/state/robotmcp/install-manifest.json`) recording per agent+scope: files touched, created-whole-file vs inserted-key, and the hash of the written value.
- [x] 4.2 Manifest read/write is atomic and forward-compatible (schema version field); `install` updates it, `uninstall` consumes it, `list` reads it.

## 5. Documentation & release

- [x] 5.1 README "Install into your coding agent" section: extras matrix, `uv tool install "rf-mcp[all]"` → `robotmcp init` → `robotmcp install`, the supported-agent table, and the manual `{"command":"robotmcp"}` fallback.
- [x] 5.2 Document scope defaults (project where supported, explicit `--scope user` for global) and the `--dry-run`/`uninstall` safety story.
- [x] 5.3 Bump `version` in `pyproject.toml` (0.31.2 → 0.33.0). NOTE: the PyPI publish itself is a release-time action requiring credentials/CI — not performed here.

## 6. Tests

- [x] 6.1 Per-adapter round-trip fixtures: sample existing config (JSON/JSONC/TOML/YAML) → `install` inserts the `robotmcp` server → parse-back asserts the correct shape AND that pre-existing servers/keys survive.
- [x] 6.2 `uninstall` on a manifest entry whose value is unchanged removes it; a user-edited entry is preserved and reported; `--dry-run` writes nothing.
- [x] 6.3 Detection unit tests (monkeypatched config dirs/binaries); `--agents detected` vs `all` vs `<csv>` selection.
- [x] 6.4 `init`/`doctor`/`--version` unit tests (mock subprocess for Browser.entry; monkeypatch `find_spec`; `--version` == `importlib.metadata.version`).
- [x] 6.5 Guard test: `python -m Browser.entry --help` succeeds in a `[web]` env (catches upstream entry-point rename); skip when Browser absent.
- [x] 6.6 Packaging smoke (CI, opt-in): build wheel, `uv tool install "<wheel>[api]"` in a clean env, assert `robotmcp` on PATH and MCP handshake returns tools — mirrors `experiments/uv-tool-install/probe_toolinstall.sh`.
