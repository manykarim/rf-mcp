#!/bin/bash
# Desktop harness entrypoint (change: desktop-docker-agent-harness).
#
# Brings up, in order, a session D-Bus -> Xvfb :99 -> fluxbox (EWMH WM) ->
# the AT-SPI accessibility bus (launcher + registryd) -> x11vnc/noVNC, then
# exec's the requested command ("$@", default: the deterministic smoke).
#
# A sequential entrypoint (not supervisord) is used deliberately: the AT-SPI
# stack needs a SHARED DBUS_SESSION_BUS_ADDRESS and an ORDERED bring-up
# (Xvfb ready before at-spi, dbus before at-spi) — both are fragile to express
# as independent supervisord programs. change-note: design D3 (deviation).
set -euo pipefail

log() { echo "[entrypoint] $*" >&2; }

# 1) session D-Bus (advertises the a11y bus on it).
if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
    eval "$(dbus-launch --sh-syntax)"
    export DBUS_SESSION_BUS_ADDRESS DBUS_SESSION_BUS_PID
    log "dbus session at ${DBUS_SESSION_BUS_ADDRESS}"
fi

# 2) Xvfb :99, wait until the display answers.
Xvfb "${DISPLAY}" -screen 0 "${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}x${DISPLAY_DEPTH}" -ac +extension RANDR >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!
# Ownership corroboration for the safety guard (change:
# desktop-isolation-marker-hardening): record the PID of the X server WE
# launched so the marker (ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY) can be verified
# against a live X server for this display, not trusted on assertion alone.
export ROBOTMCP_PLATYNUI_ISOLATED_XPID="${XVFB_PID}"
for _ in $(seq 1 40); do
    if xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then break; fi
    sleep 0.25
done
xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 || { log "FATAL: Xvfb ${DISPLAY} did not come up"; cat /tmp/xvfb.log >&2; exit 1; }
log "Xvfb ${DISPLAY} ready (pid ${XVFB_PID}, marked isolated)"

# 3) EWMH window manager (provides WindowSurface / focus / dialog modality).
fluxbox >/tmp/fluxbox.log 2>&1 &
sleep 0.5
log "fluxbox started"

# 4) AT-SPI accessibility bus. Debian ships the helpers under /usr/libexec.
ATSPI_LAUNCHER="$(command -v at-spi-bus-launcher || echo /usr/libexec/at-spi-bus-launcher)"
ATSPI_REGISTRYD="$(command -v at-spi2-registryd || echo /usr/libexec/at-spi2-registryd)"
if [ -x "${ATSPI_LAUNCHER}" ]; then
    "${ATSPI_LAUNCHER}" --launch-immediately >/tmp/atspi-launcher.log 2>&1 &
    sleep 0.5
fi
if [ -x "${ATSPI_REGISTRYD}" ]; then
    "${ATSPI_REGISTRYD}" >/tmp/atspi-registryd.log 2>&1 &
    sleep 0.5
fi
log "AT-SPI launcher=${ATSPI_LAUNCHER} registryd=${ATSPI_REGISTRYD}"

# 5) VNC + noVNC for live observation (best-effort; never fatal).
if command -v x11vnc >/dev/null 2>&1; then
    x11vnc -display "${DISPLAY}" -forever -shared -rfbport 5900 -nopw -quiet >/tmp/x11vnc.log 2>&1 &
fi
if command -v websockify >/dev/null 2>&1 && [ -d /usr/share/novnc ]; then
    websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
fi

log "desktop up — exec: $*"
exec "$@"
