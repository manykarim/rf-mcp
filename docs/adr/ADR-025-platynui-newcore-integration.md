# ADR-025: PlatynUI New-Core Integration (Rust core + CLI tool)

**Status:** Implemented (2026-06-06; experiments validated live)

**Implementation notes:**
- Desktop gates use `session.is_desktop_session() is True` (strict bool) so
  spec'd MagicMock sessions in existing tests don't accidentally match.
- `desktop_exec` tool profile preset already existed on main (ADR-021 P3) —
  verified wired in `services.py` preset registry and domain routing.
- `ui_tree` section implemented natively via `platynui_native.Runtime` in a
  worker thread (`ui_tree_service.py`), not via platynui-cli subprocess —
  app list always; subtree expansion only for apps named in
  `elements_of_interest`, bounded by depth=3/children=50/nodes=200.
**Date:** 2026-06-05
**Branch:** `feature/platynui_newcore_support`
**Supersedes:** ADR-012 (old-core PlatynUI integration — explicitly NOT ported; fresh implementation)

## Context

PlatynUI (`https://github.com/imbus/robotframework-PlatynUI`, branch `new_core`) was rewritten
around a Rust workspace with:

- A shared Rust UI model (`control`/`item`/`app`/`native` namespaces) + XPath 2.0-inspired engine
- Native providers: Windows UIA, Linux AT-SPI2 (macOS AX planned), mock providers (compile-gated)
- Platform device layers: `platform-linux` (mediator) → `platform-linux-x11` / `platform-linux-wayland`
- `platynui-native` PyO3 bindings (abi3-py312) — published on PyPI as dev builds
- `platynui-cli` diagnostic CLI — published on PyPI as dev builds
- A Robot Framework library `PlatynUI.BareMetal` (24 keywords, dynamic-core), **not yet on PyPI**

The decision (per user) is to ignore the old ADR-012 integration entirely and build fresh
support for the new core + CLI in rf-mcp.

## Experimental Findings (all verified live, 2026-06-05)

### E1 — Installation
- `uv pip install --pre platynui-cli platynui-native` works (0.12.0.dev330) — no source build needed for the runtime/CLI.
- `robotframework-PlatynUI` (the RF keyword library) is **not on PyPI** for the new core
  (PyPI's `robotframework-platynui` ≤0.9.2 is the old generation). Source install required.
- **Version skew hazard**: source RF library imports `WindowSurface` from `platynui_native`;
  PyPI `dev330` wheel predates it → `ImportError`. RF library and native wheel **must come
  from the same source commit** until upstream publishes matched packages.
- Python: PlatynUI needs ≥3.12 (abi3-py312). rf-mcp supports ≥3.10 → PlatynUI stays an
  **optional** integration, never a hard dependency.

### E2 — Wayland portal hang (CRITICAL)
- On a Wayland session, `Runtime::new()` synchronously initializes platform modules. On
  GNOME/KDE the Wayland input backend does a `org.freedesktop.portal.RemoteDesktop`
  handshake whose `wait_for_response` **blocks forever** (no timeout in `portal.rs`),
  and the first run requires an **interactive consent dialog**. Headless/MCP contexts hang.
- A restore-token is persisted at `$XDG_DATA_HOME/platynui/portal-restore-token` after first
  consent (persist_mode=2), suppressing later dialogs — but first-run consent is unavoidable
  on the Wayland path.
- **Fix validated**: `XDG_SESSION_TYPE=x11` (+ ensure `DISPLAY` set; unset `WAYLAND_DISPLAY`
  not required — `XDG_SESSION_TYPE` is authoritative, tested upstream in `session.rs`).
  Routes to the X11/XWayland backend: full keyboard (XTest, ~140 named keys + dynamic
  keycode remapping — the old Linux keyboard stub is gone), pointer, screenshot, EWMH
  window management. `info` runs in ~10ms, `Runtime()` in ~50ms.

### E3 — Query performance (CRITICAL)
- Desktop-wide descendant queries are pathological: `//control:Window` ≈ **47 s** on a
  14-app desktop (AT-SPI per-node 1 s timeout × sluggish nodes, Chrome subtree dominates).
- Scoped queries are fast: `/app:*` (app list) ≈ 1 s; child-axis ≈ 7 ms;
  `/app:*[@Name="gnome-text-editor"]//control:*` ≈ 1.5 s (1934 nodes).
- **Integration rule: never emit unscoped `//` queries.** Locator guidance, ui_tree
  retrieval, and generated suites must use app-scoped paths
  (`/app:*[@Name='X']//control:Button[@Name='OK']`) or `Set Root`.

### E4 — Role mapping on Linux
- GNOME/GTK app windows expose AT-SPI role **Frame** → `control:Frame`, NOT `control:Window`.
  `//control:Window` matched only gnome-shell internals. Window-management keywords work on
  Frame/Window/Dialog (window-surface pattern), but **element tests must use the right role**.
- Locator guidance must teach: Linux windows = `control:Frame` (or pattern-based selection);
  Windows UIA = `control:Window`.

### E5 — Keyword surface (24 keywords, BareMetal, scope=SUITE)
Pointer: `Pointer Click/Multi Click/Press/Release/Move To`, `Get Pointer Position`.
Keyboard: `Keyboard Type/Press/Release` (sequence syntax `<Ctrl+A>Hello`).
Window: `Activate Window`, `Maximize Window`, `Minimize Window`, `Restore Window`,
`Close Window`, `Move Window`, `Resize Window`, `Move And Resize Window`, `Bring To Front`.
Query/Read: `Query` (only_first/root args), `Get Attribute` (assertion-engine enabled),
`Set Root`, `Focus`, `Take Screenshot` (EMBED supported), `Highlight`.
- Locators are `UiNodeDescriptor` (XPath string or UiNode), lazy-resolved with
  default 30 s timeout / 0.1 s retry.
- Import args: `keyboard_profile`, `pointer_settings`, `pointer_profile`, `use_mock`,
  `auto_activate`. `use_mock=True` **raises ProviderError on stock PyPI wheels**
  (mock provider is compile-gated; not linked in published wheels).

### E6 — CLI surface (11 subcommands)
`list-providers`, `info`, `query`, `snapshot`, `watch`, `highlight`, `screenshot`, `focus`,
`window`, `pointer`, `keyboard`. JSON output: `info|query|list-providers|keyboard list
--format json`; XML: `snapshot --format xml [--max-depth N]`. Exit code 1 on error.
Every invocation constructs `Runtime` → same Wayland-hang caveat applies; CLI subprocesses
must receive the X11-forcing env.

## Decisions

1. **Fresh plugin, no ADR-012 port.** New `plugins/builtin/platynui_plugin.py` targeting the
   24-keyword new-core surface and descriptor locator model.
2. **Session-type env shim.** When a desktop session initializes PlatynUI on Linux and
   `XDG_SESSION_TYPE=wayland` (or unset with `WAYLAND_DISPLAY` present) and `DISPLAY` is
   available, rf-mcp sets `XDG_SESSION_TYPE=x11` in the MCP server process env **before**
   first `Runtime` creation, with an opt-out env var `ROBOTMCP_PLATYNUI_KEEP_WAYLAND=1`.
   Rationale: unbounded portal block + interactive consent are incompatible with MCP servers.
   The shim is logged loudly and surfaced in session init response hints.
3. **Scoped-query policy.** All rf-mcp-generated queries (ui_tree, guidance examples,
   intent transformations) are app-scoped. `get_locator_guidance` gains a
   `platynui_locators` topic (cookbook: app-scoped paths, control:Frame on Linux,
   ActivationPoint/Bounds, descriptor retry semantics, keyboard sequence syntax).
4. **ui_tree via native Runtime with depth/scope limits** (primary) and
   `platynui-cli snapshot --format xml --max-depth N` (fallback/diagnostic). Default:
   list `/app:*` (≈1 s), expand the target app's window frames one level; never full-desktop.
5. **Desktop executor skips** (same philosophy as ADR-012, re-implemented): no browser
   pre-validation, no timeout injection, no DOM page source for DESKTOP_TESTING sessions.
6. **Version-matching check.** Plugin verifies `PlatynUI.BareMetal` imports cleanly; on
   `ImportError` mentioning platynui_native symbols, returns a structured hint explaining
   the matched-set requirement (build native from the same source commit).
7. **Mock-free unit testing.** Because stock wheels lack the mock provider, rf-mcp unit
   tests mock at the Python boundary (`platynui_native.Runtime`), call-shape style
   (lesson from ADR-023). Real-desktop integration tests are skip-marked.
8. **Timeout classification.** 24 keywords mapped: pointer/window ops → CLICK,
   keyboard → FILL, Query/Get Attribute/Get Pointer Position → GET_TEXT,
   Take Screenshot → SCREENSHOT, Highlight/Set Root/Focus → CLICK-class fast ops.
9. **Intent mappings onto the EXISTING verb set only** (refined during implementation):
   CLICK→Pointer Click, FILL→Keyboard Type, HOVER→Pointer Move To,
   ASSERT_VISIBLE/WAIT_FOR→Query(only_first=True), EXTRACT/EXTRACT_TEXT→Get Attribute
   (mode=text reads `Name`). The branch's intent design keeps the verb set
   deliberately small (decision entropy for small LLMs) — window-management
   keywords are called via `execute_step` directly, NOT new verbs. NAVIGATE and
   SELECT have no PlatynUI mapping. `LocatorStrategy.PLATYNUI_XPATH` added; the
   locator normalizer passes PlatynUI descriptors through unchanged
   (`platynui_xpath_pass_through`).
10. **Tool profile** `desktop_exec` includes `build_test_suite` (P3 lesson from ADR-021).

## Architecture (integration points)

| Concern | File(s) | Action |
|---|---|---|
| Plugin | `plugins/builtin/platynui_plugin.py` (NEW), `plugins/builtin/__init__.py` | 24 keyword defs, library config, session defaults, env shim hook |
| Env shim | plugin + `components/execution/session_manager.py` | set `XDG_SESSION_TYPE=x11` pre-import on Linux/Wayland |
| Detection | `utils/library_detection.py`, `models/session_models.py` | DESKTOP_TESTING type/profile, detection patterns |
| Normalization | `server.py` `_LIBRARY_NAME_ALIASES` | "PlatynUI" → "PlatynUI.BareMetal" |
| Registry | `utils/library_registry.py` | DESKTOP category entry, install info (matched-set note) |
| Intent | `domains/intent/{value_objects,aggregates,services}.py` | verbs, 15+ mappings, PLATYNUI_XPATH strategy |
| Timeout | `domains/timeout/keyword_classifier.py` | 24-keyword classification sets |
| Tool profile | `domains/tool_profile/aggregates.py` | `desktop_exec()` preset |
| Executor | `components/execution/keyword_executor.py` | desktop skips (pre-validate/timeout/verifier) |
| Page source | `components/execution/page_source_service.py` | None entries for PlatynUI |
| ui_tree | `server.py` get_session_state | scoped `ui_tree` section |
| Guidance | locator guidance service | `platynui_locators` topic |

## Test & Benchmark Strategy

- Unit: plugin metadata, detection, intent, timeout, normalization, env shim logic
  (patched `os.environ`/platform), guidance topic content.
- Call-shape: mock `platynui_native.Runtime` at module boundary; assert exact call args.
- Integration (skip-marked, requires display): Runtime init, scoped query timing,
  BareMetal keyword execution through MCP `execute_step`, suite generation.
- Benchmarks: plugin lookup, intent resolution, ui_tree assembly latency budget
  (<50 µs added per execute_step on non-desktop paths; desktop ui_tree ≤2 s scoped).

## Risks

- Upstream is "preview quality" — keyword surface may change; pin instructions to a commit.
- PyPI dev wheels lag source; matched-set requirement is user-visible friction until a
  stable release.
- AT-SPI perf varies wildly with desktop load; guidance + scoping mitigates, cannot fix.
- Wayland-native-only environments (no XWayland) cannot use the X11 shim; portal consent
  flow with restore-token is the only path (documented, not automated).
