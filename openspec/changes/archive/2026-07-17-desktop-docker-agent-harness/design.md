# Design: desktop-docker-agent-harness

## Context

Existing `docker/Dockerfile.vnc` + `supervisord.conf` already run Xvfb `:99`, fluxbox (EWMH WM), x11vnc `:5900`, noVNC `:6080`, and robotmcp (HTTP). Base is `python:3.12-slim` (Debian 12, glibc 2.36 — compatible with the dev330 `manylinux_2_34` native wheel). Missing for desktop automation: PlatynUI (not in pyproject/extras), the AT-SPI stack (only `libgtk-3-0`), a desktop AUT (only xterm), agent wiring to a MiniMax model, and the isolation marker. Docker 29.3.1 is available; gnome-calculator exists on the host.

Blocker-dissolution the container buys (validated by the host rounds' failures):

```
  Wayland input block / screenshot stub / wayland-0 escape  → gone (pure X11)
  shared AT-SPI bus → ProcessId=gnome-shell                  → gone (container-local bus)
  active-desktop guard refusal                              → :99 is isolated → allowed
  no-WM dialog focus / missing WindowSurface                → fluxbox provides EWMH
  unscoped // walks 16 host apps (36s)                       → ~1 app → fast
```

## Goals / Non-Goals

**Goals:** a reproducible image where PlatynUI reads the AUT's AT-SPI tree, XTest input lands, screenshots work, and a deterministic smoke proves steering + read-back without any LLM; plus opencode+MiniMax wiring so an agent can drive it when a key is supplied.

**Non-Goals:** running a paid MiniMax agent loop in CI (key-gated, manual); the LibreOffice save round-trip as an acceptance gate (it's rung 4, aspirational — the gate is the calculator); changing rf-mcp `src/` behavior; GPU/hardware accel; multi-arch publish (amd64 first, arm64 best-effort since the native wheel ships aarch64).

## Decisions

### D1 — Dedicated `Dockerfile.desktop`, not an extension of the browser image
A separate image keeps the browser e2e image lean and the desktop concerns explicit. It may share the base apt layer conceptually but installs the desktop-specific stack: `at-spi2-core`, `xfonts-base`, fluxbox, `dbus-x11`, `x11-utils` (xprop/xwininfo), `xdotool` (diagnostics only), gnome-calculator, a GTK text editor (`gnome-text-editor` or `mousepad`), x11vnc/noVNC/websockify, supervisor.
*Alternative rejected*: bolting desktop onto `Dockerfile.vnc` — couples browser + desktop lifecycles and bloats both.

### D2 — PlatynUI via pinned prebuilt wheels (the dev330 finding, applied)
Install `platynui-native==<pinned dev>` and `robotframework-PlatynUI==<same>` from PyPI (prebuilt `manylinux_2_34` — no Rust toolchain in the image). Pin RF-lib and native to the **same** dev release to avoid the "ImportError for native symbols (WindowSurface)" skew. This replaces the host's `PYTHONPATH`-to-checkout hack entirely.
*Alternative rejected*: building the Rust runtime from source in the image — slow, needs cargo, and unnecessary now that wheels exist.

### D3 — AT-SPI bring-up is the central risk; make it an explicit, verified step
The AT-SPI accessibility bus is the #1 failure mode (host reports' hangs on `Runtime().evaluate_single` / UNO resolve smell like a11y-bus stalls). Bring-up sequence, ordered under supervisord:
1. a session `dbus-daemon` (dbus-x11) → `DBUS_SESSION_BUS_ADDRESS`.
2. `Xvfb :99` → `fluxbox`.
3. `at-spi-bus-launcher` + `at-spi2-registryd` on the session bus; env `GTK_A11Y=1`, `GNOME_ACCESSIBILITY=1`, `NO_AT_BRIDGE=0`, `QT_ACCESSIBILITY=1`.
4. AUT launched with a11y enabled.
Acceptance probe (in the smoke): `platynui-cli list-providers` must show **AT-SPI2 active**, and `/app:*` must resolve the AUT. If registryd isn't up, providers list is empty → the smoke fails loudly with a diagnostic, not a 30-s hang.
*Alternative rejected*: assuming apt-install is enough — the package doesn't auto-start registryd in a headless container.

### D4 — gnome-calculator is the acceptance AUT (deterministic, readable)
Calculator has no save dialogs (sidesteps the WM dialog-focus hard case), stable button names, and an AT-SPI-readable result display. The smoke: launch → `Set Root` to the calc app → drive `7 × 6 =` via button `Pointer Click` (or `Keyboard Type` with focus) → **`Get Attribute` / `Query` the result display and assert it reads `42`** → `Take Screenshot` to the artifacts dir. This is the "read respective data and ui controls" confirmation, LLM-free and CI-able. A text editor (type → select → read-back) is the rung-3 secondary target.
*Alternative rejected*: LibreOffice as the gate — dialogs + UNO make it flaky; keep it as the aspirational rung 4.

### D5 — Agent transport: opencode spawns robotmcp over stdio (Topology A)
Inside the container, opencode launches robotmcp via stdio (matching `opencode.json` today), inheriting `DISPLAY=:99` + the isolation marker + X11 pins. Simpler than the HTTP topology (no port plumbing, one env). The supervisord HTTP robotmcp stays available for the frontend/observation but the agent uses its own stdio server. MiniMax is added as an opencode provider (OpenAI-compatible endpoint or via OpenRouter) with `MINIMAX_API_KEY` from env/secret; model chosen for reliable tool-calling.
*Alternative rejected*: agent-over-HTTP-to-supervisord-robotmcp — more moving parts, and MCP-over-HTTP client config in opencode is heavier than stdio.

### D6 — Verification + artifacts
Screenshots go to `ROBOTMCP_SCREENSHOT_DIR=/artifacts` (a mounted volume) so they survive the container and satisfy the screenshot-path guard. Live observation via noVNC `:6080`. Host backstop: `docker exec <c> import -window root /artifacts/host.png`. The smoke writes a JSON result (provider status, resolved AUT, read-back value, screenshot paths) as the machine-checkable finding record.

### D7 — Staged rungs = the task ladder
```
  0 build → AT-SPI provider up (platynui-cli list-providers)
  1 launch gnome-calculator → resolve → read result display   ← DETERMINISTIC GATE
  2 opencode+MiniMax drives the calculator, asserts 42          ← key-gated
  3 text editor: type → format → read-back
  4 LibreOffice round-trip (type→save→close→reopen→assert)      ← aspirational
```
Rungs 0–1 are the acceptance gate (no key). 2–4 are runnable with a key and documented.

## Risks / Trade-offs

- [AT-SPI registryd doesn't come up headless] → explicit supervisord program + smoke probe that fails loudly; documented `busctl`/env checklist (D3).
- [dev wheel version drift / RF-lib↔native skew] → pin both to one dev release; smoke asserts `has_pattern`/`supported_patterns` import cleanly.
- [MiniMax tool-calling weak on small models] → pick a known-good model; the deterministic gate doesn't depend on the LLM, so harness quality is provable regardless.
- [Image size/build time] (GNOME calc pulls deps) → acceptable for a harness; use `--no-install-recommends`; cache the apt layer.
- [arm64] → native wheel ships aarch64; validate amd64 first, mark arm64 best-effort.
- [Screenshot黑/black frame under Xvfb without damage] → fluxbox + a wallpaper-less root is fine; the smoke asserts non-trivial PNG size + the calculator window bounds.

## Migration Plan

Purely additive (new docker/ + docs files). No `src/` change; rollback = delete the new files. The deterministic smoke is the regression guard for the harness itself.

## Open Questions

- Which exact dev pin — latest `platynui-native` dev (dev330 line) vs a tagged rc — resolved at implementation by checking PyPI for the newest installable pair.
- gnome-text-editor vs mousepad for the rung-3 editor (a11y richness vs image size) — pick at implementation.
