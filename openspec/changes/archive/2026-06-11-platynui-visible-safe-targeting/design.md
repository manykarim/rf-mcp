# Design: platynui-visible-safe-targeting

## Context

The LibreOffice Writer validation (docs/DESKTOP_PLATYNUI_LIBREOFFICE_VALIDATION_REPORT.md) exposed four rf-mcp gaps (I-1..I-4) and one workflow gap: the "visible on tester's screen + input confined to the test app" path exists (Xephyr mode in `scripts/platynui_desktop_bootstrap.sh` + ADR-027 isolation marker) but is invisible to agents and unverifiable through MCP tools.

**User directive**: build on the CLI tools and APIs the PlatynUI new-core (github.com/imbus/robotframework-PlatynUI, `new_core`, local checkout at `/home/many/workspace/robotframework-PlatynUI`) already ships, instead of rf-mcp-custom mechanisms.

Upstream surface relevant here (verified in the local checkout):

| Upstream API | What it gives us |
|---|---|
| `UiNode.supported_patterns()` / `has_pattern()` | Explicit detection of `WindowSurface` / `Focusable` availability (I-2 root cause: rf-mcp currently probes `get_pattern()` and swallows all exceptions at `platynui_focus.py:393-394`) |
| `Runtime.bring_to_front(node, wait_ms=…)` | Upstream restore + activate + **poll `WindowSurface.accepts_user_input()`** — verified activation; raises typed `BringToFrontError` (incl. `PatternMissing`, `Timeout`) |
| `Runtime.focus(node)` | Element focus via `Focusable` pattern; raises typed `FocusError::PatternMissing` |
| `Runtime.highlight(rects, duration_ms)` / `clear_highlight()` | On-screen overlay marking the exact element about to receive input (X11 override-redirect window) |
| `Runtime.screenshot(rect)` | Evidence capture (PNG bytes) |
| `Runtime.desktop_info()` | technology, bounds, os, monitors — display reporting for session state |
| `Application` pattern attributes (`ProcessId`, `ProcessName`, `CommandLine`) | PID-based proof that a target node belongs to the launched AUT (docs/architecture.md §6.3) |
| `platynui-cli-rs` (`window --list/--activate`, `highlight`, `screenshot`, `info`, `snapshot`) | Operator-side verification commands for recipes/docs (docs/cli.md) |

Upstream constraint (docs/architecture.md §8.5): on Linux, `WindowSurface.activate()` and `WindowManager.resolve_window()` are implemented via **EWMH** (`_NET_ACTIVE_WINDOW`, `_NET_CLIENT_LIST` + `_NET_WM_PID`) — they require an EWMH-capable window manager on the bound display. The highlight provider (override-redirect window) works without one.

Current rf-mcp focus-before-act (`platynui_focus.py`) hand-rolls three tiers: manual `WindowSurface.activate()`, legacy `runtime.bring_to_front`, and a custom ctypes `XRaiseWindow`-by-PID — with **silent** pattern-miss behavior. Input confinement today is by display isolation (ADR-027 marker + EWMH probe in `desktop_display_safety.py`): XTest input is display-scoped, so "isolated display + verified focus on the AUT window" is the correct, provable confinement model.

## Goals / Non-Goals

**Goals:**
- G1 (I-1): no silent empty suites — executed/recorded/failed accounting visible in `execute_step` and `build_test_suite`.
- G2 (I-2): focus verification rebuilt on upstream `supported_patterns()` + `bring_to_front(wait_ms)` + `accepts_user_input()`; explicit `platynui_focus_warning` when focus is unverifiable.
- G3: visible-and-confined execution — upstream `highlight()` marks the target on screen before interaction; session state proves display identity, isolation classification, and upstream `desktop_info()`; recipes recommend the visible Xephyr mode and `platynui-cli` verification commands.
- G4 (I-3): `assign_to` variables persist to session scope on the non-context path.
- G5 (I-4): detect-and-hint for Process `=`-argument misparse.

**Non-Goals:**
- Fixing upstream/app blockers: LibreOffice's missing `WindowSurface`/text exposure, gnome-calculator AT-SPI absence, Wayland native input (libei/EIS) — rf-mcp diagnoses, never works around.
- Building any custom spy/inspector/highlighter (upstream `platynui-inspector` and `runtime.highlight` exist).
- Per-window input grabs/locks (X11 has no safe primitive for this; confinement remains display-isolation + verified focus).
- Recording failed steps into generated suites (they stay excluded; they just stop being invisible).

## Decisions

### D1 — Step accounting lives on `ExecutionSession`; suite warning in `test_builder`
Add `executed_step_count`, `failed_step_count` counters to `ExecutionSession` (`session_models.py`); increment at the existing success/failure branches in `keyword_executor.py` (~1965/2059). Recorded count is derived (`len(session.steps)` + registry tests + suite-level). `execute_step` responses gain `steps_executed`/`steps_recorded` alongside the existing `recorded` flag. `build_test_suite` adds the three counts to `statistics` and, when `executed > 0` and the suite body contains only launch/setup steps (or nothing), prepends a top-level `warning` mirroring the ADR-021 P12 pattern: *"N steps executed but only M recorded — failed steps are never recorded; the generated suite may be empty. Check execute_step results."*
*Alternative rejected*: recording failed steps with a FAIL marker — changes suite semantics and breaks the "suite must be runnable" invariant.

### D2 — Focus tiers become upstream-first; pattern introspection before action
Rework `PlatynUIFocusManager.focus_window()`:
1. **Introspect**: `window.supported_patterns()` (hasattr-guarded) → stored on `FocusOutcome.patterns`.
2. **Tier 1**: `runtime.bring_to_front(window, wait_ms=cap)` — catches typed `BringToFrontError`; on success record `strategy="bring_to_front"`, and read `WindowSurface.accepts_user_input()` → `FocusOutcome.input_ready: bool|None`.
3. **Tier 2**: `runtime.focus(target_node)` when `Focusable` is supported → `strategy="focus"`.
4. **Tier 3 (last resort, demoted)**: existing ctypes `_x11_raise` → `strategy="x11_raise"`, ALWAYS paired with the unverifiable warning.
When introspection shows neither `WindowSurface` nor `Focusable` (the LibreOffice case), emit the I-2 warning verbatim: *"input focus could not be verified for this target — keystrokes may not land (no WindowSurface/Focusable pattern)"* via the existing `platynui_focus_warning` hint channel (`keyword_executor.py:1956-1963` — no new plumbing).
**PID scope check (upstream data, replaces nothing — extends ADR-026 scope guard)**: when the session launched the AUT via Process (PID known), compare it against the target's `app:Application` ancestor `ProcessId` attribute (upstream Application pattern); mismatch adds a scope warning *"resolved target belongs to PID X, but the AUT was launched as PID Y"* — explicit proof that commands go only to the test application.
*Alternative rejected*: keeping ctypes-first ordering — contradicts the upstream-first directive and skips upstream's input-readiness verification.
*Compat*: every upstream call is `hasattr`-guarded; an older `platynui_native` silently degrades to current behavior + warning, never crashes.

### D3 — Target highlighting via upstream `highlight()`, default ON, soft-fail
Before each PlatynUI interaction keyword (same gate as focus-before-act), resolve the target element bounds and call `runtime.highlight(rect, duration_ms=600)`. Controlled by session config `platynui_highlight` (default `True`) and env `ROBOTMCP_PLATYNUI_HIGHLIGHT=0` kill-switch. Strictly soft-fail (try/except, debug log); on headless Xvfb it is invisible and harmless; budget ≤ 50ms — if the highlight call ever blocks longer, the soft-fail wrapper logs and the feature can be disabled per session. The element rect comes from the already-resolved descriptor node's `Bounds` attribute (no extra XPath query when focus-before-act already resolved it).
*Alternative rejected*: screenshot-per-step evidence — too heavy for stepwise loops; `screenshot` stays available via existing Take Screenshot keyword and `platynui-cli screenshot` in recipes.

### D4 — Session state proves "visible + confined"; recipes recommend Xephyr
New `desktop_environment` block in `get_session_state` desktop sections (built in `ui_tree_service.py` / server.py): `{display, isolation: isolated|active|unknown, isolation_source: marker|ewmh|none, desktop_info: {technology, bounds, os_name, monitors[...]}}` — classification reuses `desktop_display_safety.classify()` verbatim; `desktop_info` comes from upstream `runtime.desktop_info()` (subprocess-isolation not needed: the runtime broker already binds the display). `build_isolation_recipe()` reorders to present **visible (Xephyr)** first as "recommended for interactive testing — app visible on your screen, input confined to the nested display", headless Xvfb second, and appends operator verification commands: `platynui-cli-rs info`, `platynui-cli-rs window --list`, `platynui-cli-rs highlight '<xpath>'`, `platynui-cli-rs snapshot '<xpath>'`. The safety-guard refusal message points at the same recipe.
**EWMH WM inside the nested display**: because upstream `WindowSurface.activate()`/`resolve_window` need an EWMH WM (docs/architecture.md §8.5), the visible recipe and `scripts/platynui_desktop_bootstrap.sh --mode visible` start a minimal EWMH WM (e.g. `openbox` if available) inside Xephyr; without one, focus verification degrades to the D2 warning path. ADR-027 classification is unaffected — isolation is by marker, and a WM inside the nested display does not mark it active.
*Why this proves confinement*: XTest input targets the bound `DISPLAY` only; marker-verified isolation + `desktop_info` showing the nested display's geometry + visible Xephyr window = the tester sees the app AND knows host `:0` cannot receive synthetic input.

### D5 — Non-context `assign_to` persists to `session.variables`
At the non-context assignment point (`keyword_executor.py` ~2065-2069, where `step.variables` is populated), also write the normalized `${name}` → value into `session.variables`. Identical normalization to the context path so subsequent context steps resolve them. No behavior change for the context path.

### D6 — Process `=`-arg hint: heuristic at launch sanitization + hints checker
Two hooks, one heuristic: a positional argument to `Start Process`/`Run Process` matching `^-{1,2}[^=\s]*[:.][^=\s]*=` or `^-{1,2}[^=\s]+=` (dash-prefixed, contains `=`, left side not a valid Python identifier and not a known Process config prefix like `env:`/`shell`/`cwd`/`alias`/`stdout`/`stderr`) triggers a hint: *"RF may parse '<arg>' as a named argument and drop it from the command line — escape as '<left>\\=<right>'."* Emitted (a) proactively from `_maybe_sanitize_desktop_launch()` for desktop sessions, and (b) reactively from a new `hints.py` checker when a Process launch fails. Detection only — never auto-rewrite the argument.
*Alternative rejected*: auto-escaping on the user's behalf — silently mutating launch commands is exactly the class of magic ADR-029's Run Process sanitization had to claw back.

## Risks / Trade-offs

- [`platynui_native` version skew — local build may predate `supported_patterns`/`wait_ms`] → every upstream call hasattr/try-guarded; degraded path keeps current behavior and adds the unverifiable warning; one integration test asserts graceful degradation with a stub runtime.
- [`bring_to_front(wait_ms)` adds latency per interaction step] → cap at 1500ms, only on desktop interaction keywords, only when window resolution succeeded; outcome cached per (session, window runtime_id) until `desktop_tree_dirty` (ADR-031 flag) invalidates it.
- [Highlight overlay could interfere with screenshots or flaky WMs] → 600ms duration expires before typical step cadence; `clear_highlight()` called before Take Screenshot keywords; kill-switch env + per-session flag.
- [Empty-suite warning could fire on legitimately exploration-only sessions] → warning only when ≥3 executed steps and ≥1 failed step, message states the cause neutrally; it is a `warning` field, not an error.
- [I-4 heuristic false positives (legit `env:VAR=value` kwargs)] → known-prefix allowlist; hint-only (nothing is modified), so a false positive costs one advisory line.
- [LibreOffice still fails after all this] → expected: I-2's deliverable is the *warning*, not a fix; the report's §5 blockers remain upstream.

## Migration Plan

Additive only; no API removals. Ship behind existing response-shape conventions (new optional fields). Rollback = revert; no data migration. Unit suite baseline 6665 passed + 1 skipped must stay green; new tests per capability under `tests/unit/`.

## Open Questions

- Should `FocusOutcome.input_ready == False` (window activated but `accepts_user_input()` still false after wait) escalate from warning to failure when `platynui_strict_scope` is set? Default: warning-only; strict mode escalation can ride a follow-up.
- Whether `build_isolation_recipe()` should detect an already-running Xephyr (`/tmp/platynui_bootstrap_xephyr.log` / display socket) and tailor the recipe — nice-to-have, not required for the capability.
