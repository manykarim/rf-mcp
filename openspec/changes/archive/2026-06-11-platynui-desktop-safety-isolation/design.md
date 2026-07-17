## Context

Source of record: `docs/gnome-calculator-mcp-findings-2026-06-08.md` — a real
VS Code MCP session that tried to automate GNOME Calculator stepwise and hit a
stack of failures. Reproduction during proposal authoring confirmed the most
actionable ones on the current branch:

- **Classification**: `ExecutionSession.detect_session_type_from_scenario(
  "Test the calculator app")` returns `mobile_testing` (the bare token "app"
  outweighs the desktop-app name "calculator"). Explicit "gnome calculator
  desktop" already routes to `desktop_testing`, so the gap is the ambiguous
  case the report hit.
- **Snap env present**: the shell carries snap paths
  (`/var/lib/snapd/desktop` in `XDG_DATA_DIRS`, `/snap/bin` in `PATH`); a VS
  Code snap session additionally injects snap-rooted `LD_LIBRARY_PATH`, which
  is what produces `gnome-calculator: symbol lookup error:
  /snap/core20/.../libpthread.so.0: undefined symbol __libc_pthread_init`.
- **Runtime binding**: `results/output.xml` shows `ProviderError ... Linux
  Platform failed to initialize: x11 connection: not available after shutdown
  or failed connect` even though `xdpyinfo` was healthy on `:99`.
- **Safety**: the user reported clicks/typing leaking onto the active IDE
  desktop (`DISPLAY=:0`), and step results carried "could not raise/focus the
  AUT window" (a focus warning from change `platynui-focused-execution`).

This change composes with `platynui-focused-execution` (ADR-026: focus-before-
act, ctypes `XRaiseWindow`-by-PID, visibility/scope, the focus-runtime
`clear_cache` fix) and ADR-025 (`resolve_extents` coordinate patch). The new
work is the **safety guard**, **classification fix**, **launch
sanitization**, **runtime-binding hardening**, and a **first-class isolation
bootstrap** — investigation-driven, fix the reproduced findings.

## Goals / Non-Goals

**Goals:**
- Make it impossible (by default) for a desktop session to silently drive the
  user's active desktop; provide an explicit opt-in.
- GNOME/native-desktop scenarios classify as desktop, never mobile/Appium; a
  desktop session accepts `Process`/PlatynUI.
- Desktop GUI launches survive a snap/IDE-contaminated parent environment and
  land on the bound display.
- The PlatynUI runtime binds once, after the display is settled, with no
  "not available after shutdown" on the MCP/Robot path.
- A supported isolation bootstrap + a reproduction harness + a validated
  stepwise calculator suite.

**Non-Goals:**
- Fixing snap packaging of VS Code or gnome-calculator (we sanitize the launch
  env; we do not repackage anything).
- Guaranteeing pure-Wayland-only input (X11/XWayland remains the supported
  path, per ADR-025/026).
- Re-deriving the focus/visibility behavior already shipped by
  `platynui-focused-execution` (reused, not reinvented).

## Decisions

### D1: Safety guard at the desktop chokepoint, fail-closed
Add the bound-display safety check immediately before focus/dispatch in the
executor desktop chokepoint (where `ensure_x11_session_env` + focus-before-act
already run). Decision: **fail closed** — allow without an override only when
the display is **provably isolated**; refuse on both `active` and `unknown`.
Opt-in via `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1` (and a session
attribute); bypassed runs are flagged in the payload + logged WARNING.
Rationale: one chokepoint, consistent with ADR-026; refuse-by-default is the
posture the report demands. Alternative considered: warn-only — rejected; the
report shows warnings are not enough (input still leaked).

### D2: Tri-state classification by positive isolation proof (cross-LLM review)
Both review CLIs flagged that "no EWMH WM = isolated" is unsound: it
false-allows non-EWMH real desktops and any `:0` where the probe silently
fails, and false-blocks a bootstrapped display that runs its own WM. Decision:
classify `isolated` / `active` / `unknown`, where **`isolated` requires
positive proof** (an rf-mcp isolation marker recorded by the bootstrap — a
dedicated `XAUTHORITY`/socket + display-provenance marker), `active` = EWMH
`_NET_SUPPORTING_WM_CHECK` resolves to a live window on the bound display, and
**everything else is `unknown` and refused**. The isolation marker takes
precedence over the EWMH probe (so a nested display with an internal WM is
still `isolated`). A probe error defaults to `unknown` (refuse), never
`isolated`. Rationale: the only safe rule is "prove isolation", not "fail to
find a WM". `XDG_SESSION_TYPE`/`WAYLAND_DISPLAY` are not classifier inputs (the
EWMH probe + marker are authoritative).

### D3: Launch sanitization as a Process.Start Process pre-hook
Intercept desktop-AUT launches in the executor/plugin path: when a
DESKTOP_TESTING session runs `Start Process` for a GUI binary, build a
sanitized child env = parent env minus snap-rooted loader vars
(`LD_LIBRARY_PATH`, `LD_PRELOAD`, `GTK_PATH`, `GTK_EXE_PREFIX`,
`GIO_MODULE_DIR`, `GSETTINGS_SCHEMA_DIR` entries under `/snap/`) plus the
session's display vars. Pass it through RF's `Start Process` `env=`/`env:`
mechanism. Rationale: the report proves the same binary works from a clean
shell/`uv run python` subprocess but not from the snap-contaminated MCP path;
the minimal correct fix is to clean the child env, not the server env.
Alternative considered: cleaning the whole MCP-server env at startup —
rejected as too broad (could break the server's own deps); scope to the AUT
launch.

### D4: A runtime broker; ui_tree_service is the proven proximate cause
The cross-LLM review (OpenCode, reading the source) identified the smoking
gun: `ui_tree_service._collect_ui_tree_sync` creates a fresh `pn.Runtime()` and
calls `runtime.shutdown()` in its `finally:` on **every**
`get_session_state(sections=["ui_tree"])` call — and the platform module is
process-global and not re-initializable, so a later keyword's bind fails with
"not available after shutdown". The new safety-state surfacing (which also
reads `ui_tree`) would compound this. Decision: introduce a single **runtime
broker** (a `get_runtime()` accessor next to `ensure_x11_session_env`) with
explicit states (`open`/`shutting_down`/`closed`), a lock-protected lazy first
bind (extend the existing `_ENV_SHIM_LOCK` pattern), and thread-affine access;
**all** call sites (`platynui_focus._get_runtime`,
`ui_tree_service._collect_ui_tree_sync`, BareMetal/attach where applicable) use
it. `ui_tree_service` stops calling `shutdown()` and stops creating its own
runtime. The broker refuses re-init after `closed` with a clear "restart the
process" error. Scope the broker to the bound session/display so a display
switch does not poison the process (per Codex's note). Rationale: this is the
actual root cause of finding #3; "bind once, reuse, never re-init" is only
achievable by removing the per-call create/shutdown.

### D5: Isolation bootstrap as a documented, scripted entry point
Provide `scripts/run_gnome_calculator_isolated.sh`-class tooling (and/or a
session bootstrap option) that prepares Xvfb + the documented env + a session
bus, then runs the confined session — replacing ad-hoc per-suite setup. The
existing pytest harness (`tests/integration/test_platynui_gnome_apps_e2e.py`)
is the reference; the bootstrap codifies its recipe (systemd-run launch,
GDK/GSK env, `clear_cache` freshness). Rationale: the report's maintainer
question — "should rf-mcp expose a first-class desktop isolation/bootstrap
tool instead of pushing this burden into ad hoc Robot suite setup?" — answer:
yes. Keep it a script + documented option, not a new always-on MCP tool, to
limit surface.

### D5a: Visible isolation via a nested X server (Xephyr), VNC as fallback
Provide a **visible isolated display** so users can watch stepwise
interactions without using their active session. Mechanism: a nested X server
(Xephyr) renders the isolated `:N` display as a window on the host — the
session runs confined on `:N`, while the Xephyr window lets the user observe.
**Confinement is by the positive isolation contract (D2), not by EWMH
absence**: the bootstrap records the rf-mcp isolation marker for `:N` and that
marker (not the EWMH probe) classifies it `isolated`, so a nested display that
runs its own WM is still confined. The bootstrap must set the **MCP server
process's own** `DISPLAY`/`XAUTHORITY` to `:N` before the first PlatynUI bind
(the Rust core's X connection is established by the first `Runtime()` in the
*server* process), and force every launched child's `DISPLAY` to `:N`. VNC
fallback (Xvfb + `x11vnc`) binds to localhost / a private socket by default.
**Maturity:** the visible Xephyr/VNC mode is the preferred UX for interactive/
stepwise use (the user's explicit ask) but ships as an **opt-in preview**;
headless Xvfb is the validated/CI default and the supported baseline. Caveat
documented: host global shortcuts may still intercept some key chords in
visible mode (input is confined to `:N`, but the visible window is not a fully
independent seat). Alternatives considered: (a) running on the user's real
display confined to the AUT — rejected, that is exactly the leak the safety
guard blocks; (b) screenshots-only observation — rejected, the user wants to
watch live interactions.

### D6: Reproduction harness drives the fixes (TDD-ish)
Each finding gets a reproduction (the report's Repro A–D): A (classification)
as a unit test; B (snap launch) as an env-sanitization unit test using a
synthetic snap-rooted `LD_LIBRARY_PATH`; C (Robot runtime init) compared
against the pytest recipe; D (fresh-isolated input) folded into the isolated
e2e (already covered by ADR-026 focus + ADR-025 coordinate patch — verify
under the bootstrap). Rationale: the user asked to "experiment to reproduce to
find fixes"; the harness makes the fixes verifiable and regression-proof.

## Risks / Trade-offs

- **False "active desktop" positives** block legitimate runs → Use the EWMH
  WM probe (authoritative) and the explicit opt-in; document the escape hatch
  prominently. A misdetected isolated display must never block (D2).
- **Over-aggressive env stripping breaks the AUT differently** → Strip only
  snap-rooted entries (path-prefixed `/snap/`), preserve everything else;
  unit-test the sanitizer with a synthetic contaminated env.
- **Refuse-by-default is a breaking behavioral change** → Gated on
  DESKTOP_TESTING + active-desktop detection + opt-in; web/mobile/API
  unaffected; clear error tells the user exactly how to proceed.
- **Runtime re-bind audit may reveal a deeper PlatynUI limitation** → If the
  "not available after shutdown" is purely upstream, document it + ensure we
  never trigger the re-init; file upstream.
- **VS Code snap specifics vary by version** → The sanitizer targets the
  mechanism (snap-rooted loader vars), not a VS Code version; reproduction
  uses a synthetic env so it is host-independent.

## Migration Plan

1. Reproduction harness for findings A–D (red).
2. Classification fix (A) + desktop library allow-list; unit tests green.
3. Launch sanitization (B); unit test with synthetic snap env green.
4. Safety guard (D1/D2) refuse-by-default + opt-in + bound-display state.
5. Runtime-binding audit/hardening (D4) + isolation bootstrap (D5); isolated
   stepwise calculator suite green.
6. ADR + report cross-reference; release notes for the breaking default.
7. Transition aid: ship an optional `ROBOTMCP_PLATYNUI_SAFETY_GUARD=warn` mode
   for one release that logs the would-be refusal without blocking, so users
   can validate their isolation setup before the refuse-by-default takes full
   effect. Default remains refuse (the report shows warn-only is insufficient).
8. Rollback: safety guard and sanitization are gated on DESKTOP_TESTING and
   reversible (`ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1`, the warn mode above,
   a no-sanitize escape hatch); non-desktop sessions untouched.

## Cross-LLM review (resolved)

Reviewed by Codex CLI and OpenCode (MiniMax M3); both returned
REQUEST_CHANGES with convergent must-fix items, now folded in:

- Safety classifier made **tri-state with positive isolation proof**, fail
  closed on `active`/`unknown` (D1/D2) — was an unsound "no-EWMH = isolated".
- Visible isolation given a **positive isolation contract** (marker beats EWMH
  probe, dedicated `XAUTHORITY`, server + child `DISPLAY` binding); demoted to
  opt-in preview with documented key-chord caveat (D5a).
- **Root cause of finding #3 identified**: `ui_tree_service` creates+shuts down
  a runtime per call; fixed via the runtime broker (D4, task 5.1a).
- Classification precedence made **normative** with negative cases (calculator
  app on android → mobile); sanitization **broadened** (more vars, snap-AUT
  preservation, per-segment filtering); finding #5 gets a **live mutation
  acceptance**; opt-in is **auditable**; a one-release **warn mode** aids
  migration.

## Open Questions

- Should the runtime broker be process-global or strictly per-session/display?
  (Lean: per-bound-session/display so a display switch cannot poison the
  process, per Codex; the platform module is process-global, so the broker
  guards re-init rather than allowing parallel binds.)
- Should the isolation bootstrap be a script only, or also a `manage_session`
  option (e.g. `isolation="xvfb"|"xephyr"|"vnc"`)? (Lean: script first; add the
  option if demand is clear.)
