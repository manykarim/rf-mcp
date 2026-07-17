# Proposal: desktop-suite-replay-environment

## Why

The run-4 generated suite replays end-to-end under plain `robot` — but only when the operator hand-supplies the display environment. The user's own invocation (`uv run robot -d results libreoffice_writer.robot` from a normal Wayland shell, 2026-06-12) failed with `ProviderError: Wayland screenshot provider: not yet implemented`: PlatynUI's session detection reads `XDG_SESSION_TYPE=wayland` (authoritative, upstream architecture.md §3) and initializes the unimplemented Wayland backend, and GTK AUTs would additionally escape to the host compositor via the `wayland-0` fallback. The hand-fix — a `Suite Setup` keyword pinning `DISPLAY` / `XDG_SESSION_TYPE=x11` / `GDK_BACKEND=x11` / `QT_QPA_PLATFORM=xcb` before the first PlatynUI keyword (safe: upstream `runtime` is a lazy `@cached_property`) — made the suite pass with the user's exact command. Generated desktop suites should carry this preamble themselves.

## What Changes

- `build_test_suite` for desktop (PlatynUI) sessions emits a **replay-environment preamble**: a `Prepare Desktop Display Environment` keyword in the `*** Keywords ***` section (pinning the session's bound DISPLAY, `XDG_SESSION_TYPE=x11`, `GDK_BACKEND=x11`, `QT_QPA_PLATFORM=xcb`, and removing `WAYLAND_DISPLAY`), wired as `Suite Setup`, with `OperatingSystem` added to imports.
- A user-defined suite setup (session `suite_setup`) takes precedence — the preamble keyword is still emitted in the Keywords section (callable manually) but not wired as Suite Setup, and the response notes this.
- The `DISPLAY` pin uses the session's bound display at build time; when no display is known, the DISPLAY line is omitted (the other pins remain — they are inert on pure-X11 hosts and ignored on Windows).
- Non-desktop sessions are untouched.

No breaking changes — additive output for desktop suites only.

## Capabilities

### New Capabilities

- `desktop-suite-replay-environment`: generated desktop suites are self-sufficient for standalone replay — they pin the display/backend environment before the first PlatynUI keyword.

### Modified Capabilities

(none — generation behavior is additive; existing capability specs in `openspec/specs/` are not contradicted)

## Impact

- `src/robotmcp/components/test_builder.py` — preamble injection in `build_suite` for desktop sessions: append a `BddKeyword`-shaped entry to `suite.bdd_keywords` (the renderer emits `*** Keywords ***` whenever that list is non-empty, independent of `bdd_style` — verified at the rendering site ~2888), set `suite.setup` when absent, add `OperatingSystem` to imports.
- Tests: new unit file `tests/unit/test_suite_replay_environment.py`; baseline 6795 passed + 1 skipped stays green; existing desktop suite-shape tests audited for the new Settings/Keywords lines.
