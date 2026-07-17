# Proposal: desktop-docker-agent-harness

## Why

Every LibreOffice/PlatynUI validation round to date fought the **host**, not the tooling: a Wayland `:0` session that blocks synthetic input, a session-global AT-SPI bus that reports the compositor's PID, a safety guard that (correctly) refuses the active desktop, and WM-less Xvfb hacks. A self-contained Docker container with a clean X11 desktop inverts all of it — pure X11 (XTest input + screenshot work), a container-local AT-SPI bus (correct PIDs, tiny tree), a genuinely-isolated display the guard can allow, and a real window manager so `Activate Window`/`Focus`/dialog navigation work. This is the environment PlatynUI's new-core is designed for, and it's the natural home for two things we already surfaced: the **dev330 prebuilt native wheel** (replaces the local-checkout `PYTHONPATH` hack) and the **replay-environment X11 pins** (become container defaults). ~60% of the scaffolding already exists (`docker/Dockerfile.vnc` runs Xvfb + fluxbox + x11vnc/noVNC + robotmcp); the missing pieces are desktop enablement (PlatynUI + AT-SPI + an AUT) and agent wiring (opencode + a MiniMax-class model).

## What Changes

- A reproducible **desktop harness image** (`docker/Dockerfile.desktop`) providing an isolated X11 desktop: Xvfb `:99` + an EWMH window manager (fluxbox) + the **AT-SPI accessibility stack** (`at-spi2-core`/registryd, `GTK_A11Y=1`) + **PlatynUI installed from prebuilt wheels** (`platynui-native` + `robotframework-PlatynUI`, pinned) + a desktop **AUT** (`gnome-calculator`, plus a GTK text editor) + robotmcp.
- The display is marked isolated (`ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=:99`) so the desktop safety guard allows input, and the X11 backend is pinned (`XDG_SESSION_TYPE=x11`, `GDK_BACKEND=x11`).
- A **deterministic smoke harness** (no LLM) that proves the stack end-to-end: AT-SPI provider is up → launch gnome-calculator → resolve it in the tree → drive buttons → **read the result display back via a keyword** and confirm via **screenshot** — the "steering actions work and are confirmed visually or by control read-back" acceptance test.
- **Agent wiring**: an opencode configuration with a MiniMax provider (API key via secret/env) talking to robotmcp over stdio inside the container, plus a run script and an example prompt (automate the calculator; confirm the result). The LLM-driven rungs are runnable when a key is supplied; the deterministic smoke needs none.
- Observation channels documented: `Take Screenshot` artifacts (to a mounted dir), live noVNC `:6080`, and a host `docker exec` screenshot backstop.

## Capabilities

### New Capabilities

- `desktop-docker-agent-harness`: a container image + smoke harness + agent wiring that lets a coding agent automate a desktop application on an isolated in-container X11 desktop, with screenshot and keyword-read-back verification, reproducibly.

### Modified Capabilities

(none — additive; the existing `Dockerfile.vnc` browser image is untouched)

## Impact

- `docker/Dockerfile.desktop` (new), `docker/supervisord.desktop.conf` (new): Xvfb + fluxbox + AT-SPI registryd + robotmcp; PlatynUI prebuilt-wheel install; gnome-calculator + editor.
- `docker/desktop_smoke.sh` (new): the deterministic rung-0/1 acceptance test (providers up → launch → query → read-back → screenshot).
- `docker/opencode.minimax.json` + `docker/run_agent.sh` (new): agent wiring + example prompt (rungs 2+, key-gated).
- `docs/` (new): a short "desktop automation in Docker" runbook (build, smoke, agent, observe, artifacts).
- Optional `.github/workflows/` hook: the deterministic smoke can run in CI (no key); the agent run is manual/keyed. No change to existing web e2e.
- No `src/` runtime changes expected — the harness consumes existing rf-mcp desktop behavior (isolation marker, guards, screenshot path policy honored via `ROBOTMCP_SCREENSHOT_DIR`).
