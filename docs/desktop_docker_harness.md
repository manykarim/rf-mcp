# Desktop Automation in Docker — PlatynUI + robotmcp + agent

A reproducible, isolated **X11 desktop in a container** for driving real desktop
applications through robotmcp + PlatynUI, with screenshot and keyword read-back
verification. change: `desktop-docker-agent-harness`.

## Why a container

Every host validation round fought the environment, not the tooling. A container
with a clean X11 desktop dissolves those blockers:

| Host blocker (validation rounds) | In this container |
|---|---|
| Wayland blocks synthetic XTest input | gone — pure X11, input lands |
| Wayland screenshot provider stubbed | gone — X11 screenshot works |
| Shared AT-SPI bus → `ProcessId` = gnome-shell | gone — container-local bus, correct PIDs |
| Safety guard refuses the active desktop | `:99` is isolated → marked & allowed |
| GDK `wayland-0` escape to host compositor | gone — no Wayland socket exists |
| No WM → dialog focus fails, no `WindowSurface` | fluxbox (EWMH) → `Activate Window`/`Focus` work |
| Unscoped `//` walks 16 host apps (36 s) | ~1 app → fast |

## What's inside

`docker/Dockerfile.desktop` (base `python:3.12-slim`):

```
  entrypoint.sh brings up →  dbus session
                             Xvfb :99  (1280×1024×24)
                             fluxbox   (EWMH window manager)
                             at-spi-bus-launcher + at-spi2-registryd  (accessibility)
                             x11vnc :5900 + noVNC :6080  (live view)
  installed →  robotmcp (from src) + PlatynUI 0.12.0.dev330 prebuilt wheels
               (platynui-native + robotframework-PlatynUI + platynui-cli)
  AUT →  gnome-calculator, mousepad
  env →  ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=:99, XDG_SESSION_TYPE=x11,
         GDK_BACKEND=x11, GTK_A11Y=atspi, ROBOTMCP_SCREENSHOT_DIR=/artifacts
```

PlatynUI is installed from **pinned prebuilt wheels** (no in-image Rust build),
RF-lib and native pinned to the same dev release to avoid symbol skew.

## Build

```bash
docker build -f docker/Dockerfile.desktop -t robotmcp-desktop .
```

## Rung 0–1: the deterministic smoke (no LLM — the acceptance gate)

```bash
mkdir -p artifacts
docker run --rm -v "$PWD/artifacts:/artifacts" robotmcp-desktop
```

The default CMD runs `desktop_smoke.sh` → `desktop_smoke_driver.py`, which:
0. asserts an **AT-SPI2 provider is active** (fails fast, never hangs, if not);
1. launches gnome-calculator, resolves it, confirms `supported_patterns()` imports;
2. **brings it to front** (exercises `WindowSurface` via fluxbox), types `7*6=`,
   and **reads the result display back** — asserting the value is `42`;
3. writes `artifacts/calc.png` and `artifacts/smoke_result.json`.

Exit 0 = PASS. Inspect `artifacts/smoke_result.json` for the machine-checkable
findings (provider status, resolved AUT, read-back value, screenshot bytes).

## Continuous integration (gated)

The deterministic smoke runs in CI as its own **`.github/workflows/desktop-smoke.yml`** workflow
(change: `desktop-smoke-ci`). It is **keyless** (no model credential) and **gated** — it triggers on
`schedule` (weekly) + `workflow_dispatch` only, never on push/PR, because the image is large to build
and PlatynUI is pinned to a pre-release wheel, so a break surfaces on a cadence instead of red-ing
every push. A dedicated workflow (rather than a job inside `e2e-weekly.yml`) lets it be dispatched in
isolation — verifying it doesn't trigger the heavier model-driven jobs. The job:

```
docker build -f docker/Dockerfile.desktop -t robotmcp-desktop .
docker run --rm -v "$PWD/artifacts:/artifacts" robotmcp-desktop   # gates on exit 0
```

On failure it uploads `artifacts/*.png` + `artifacts/smoke_result.json` as build artifacts — the
screenshot and machine-checkable record are how you diagnose a red desktop with no live display.
Trigger it manually with **Actions → Desktop Smoke → Run workflow**, or `gh workflow run
desktop-smoke.yml --ref <branch>`. A repo-root `.dockerignore` keeps the build context at ~7 MB (the
working tree is ~2.4 GB, mostly `.venv/`).

This smoke is the reachable desktop CI coverage today; running the platynui *pytest* tests or the
agenteval desktop `.robot` ports in this image are sequenced follow-ons (they need an app-launch
fixture and/or a `systemd-run`→direct-launch seam).

## Rung 2+: the agent (key-gated)

```bash
docker run --rm \
  -e MINIMAX_API_KEY=sk-... \
  -e MINIMAX_BASE_URL=https://api.minimaxi.com/v1 \
  -v "$PWD/artifacts:/artifacts" \
  robotmcp-desktop /app/docker/run_agent.sh
```

`run_agent.sh` installs opencode on demand, points it at MiniMax + robotmcp
(stdio, `DISPLAY=:99`), and runs the example prompt (open the calculator, compute
7×6, confirm via screenshot **and** by reading the result display). Without a key
it skips gracefully — the deterministic smoke is the real gate. Edit
`docker/opencode.minimax.json` (`MINIMAX_MODEL`/`MINIMAX_BASE_URL`) for your
MiniMax account/region; the model must support tool-calling.

## Observe live

```bash
docker run --rm -p 6080:6080 -v "$PWD/artifacts:/artifacts" \
  robotmcp-desktop sleep infinity      # then open http://localhost:6080/vnc.html
# in another shell: docker exec -it <container> /app/docker/desktop_smoke.sh
```

Host-side screenshot backstop: `docker exec <c> import -window root /artifacts/host.png`.

## AT-SPI bring-up checklist (the #1 gotcha)

If the smoke reports "no AT-SPI2 provider active", check inside the container:
- `echo $DBUS_SESSION_BUS_ADDRESS` is set (entrypoint launches dbus);
- `GTK_A11Y=atspi`, `NO_AT_BRIDGE=0` are in the app's env;
- `pgrep -a at-spi2-registryd` shows the registry running;
- `platynui-cli list-providers` lists AT-SPI2 as active;
- the AUT was launched *after* the a11y bus was up.

## Isolation-marker corroboration (safety guard)

The active-desktop safety guard classifies `:99` as `isolated` from the marker
`ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=:99`. As of `desktop-isolation-marker-hardening`
the marker is **corroborated** against `ROBOTMCP_PLATYNUI_ISOLATED_XPID` — the
PID of the Xvfb the entrypoint launched — so a stale or hand-misconfigured
marker cannot false-allow input onto a real desktop. The entrypoint exports the
XPID automatically; if you launch your own isolated display, set both the marker
and `ROBOTMCP_PLATYNUI_ISOLATED_XPID=<x-server-pid>` (the bootstrap script does
this). A marker with an XPID that no longer matches a live X server for the
display fails closed (`unknown` → refused); a legacy marker with no XPID is
still honored but the session state records `isolation_source=marker_over_active_wm`
when a live WM contradicts it.

## Rung ladder

```
  0 build → AT-SPI provider up               (CI-safe, no key)
  1 launch → resolve → read result = 42       (CI-safe, no key) ← acceptance gate
  2 MiniMax agent drives the calculator        (key-gated)
  3 text editor: type → format → read-back     (key-gated)
  4 LibreOffice round-trip (aspirational)      (key-gated, dialogs are hard)
```
