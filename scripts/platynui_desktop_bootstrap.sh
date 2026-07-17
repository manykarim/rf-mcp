#!/usr/bin/env bash
#
# PlatynUI desktop isolation bootstrap (change: platynui-desktop-safety-isolation).
#
# Prepares a CONFINED, rf-mcp-owned isolated display and runs a command inside
# it with the documented environment, so desktop automation never touches the
# user's active session. Records the isolation MARKER
# (ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY) the active-desktop safety guard
# validates, and a dedicated XAUTHORITY.
#
# Modes:
#   headless   (default, CI)     : Xvfb — no on-host window
#   visible    (interactive)     : Xephyr nested X server — a window on the
#                                  host showing the isolated display so you can
#                                  WATCH stepwise interactions (still confined).
#   vnc        (headless+observe): Xvfb + x11vnc bound to localhost.
#
# Usage:
#   scripts/platynui_desktop_bootstrap.sh [--mode headless|visible|vnc]
#       [--display :N] [--geometry 1280x1024x24] [--] <command> [args...]
#
# The <command> runs with DISPLAY/XAUTHORITY/marker exported. If omitted, the
# bootstrap prints the export block and waits (Ctrl-C to tear down).
set -euo pipefail

MODE="headless"
DISPLAY_ID="${ROBOTMCP_ISOLATION_DISPLAY:-:99}"
GEOMETRY="1280x1024x24"
VNC_PORT="5999"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --display) DISPLAY_ID="$2"; shift 2;;
    --geometry) GEOMETRY="$2"; shift 2;;
    --vnc-port) VNC_PORT="$2"; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done

command -v Xvfb >/dev/null || { echo "Xvfb required" >&2; exit 1; }
command -v xdpyinfo >/dev/null || { echo "xdpyinfo required" >&2; exit 1; }

XAUTH="$(mktemp -t platynui-xauth-XXXXXX)"
PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  rm -f "$XAUTH" 2>/dev/null || true
}
trap cleanup EXIT

# Dedicated, cookie-protected XAUTHORITY for the isolated server.
COOKIE="$(mcookie 2>/dev/null || head -c16 /dev/urandom | xxd -p)"
xauth -f "$XAUTH" add "$DISPLAY_ID" . "$COOKIE" 2>/dev/null || true

start_xvfb() {
  Xvfb "$DISPLAY_ID" -screen 0 "$GEOMETRY" -auth "$XAUTH" -ac \
      >/tmp/platynui_bootstrap_xvfb.log 2>&1 &
  PIDS+=("$!")
}

case "$MODE" in
  headless)
    start_xvfb
    ;;
  visible)
    command -v Xephyr >/dev/null || { echo "Xephyr required for visible mode" >&2; exit 1; }
    # Xephyr is itself an X client of the host; it renders :N as a host window.
    # The session runs on :N (confined); the marker (not EWMH) makes :N
    # 'isolated' for the safety guard even though Xephyr may run a WM.
    Xephyr "$DISPLAY_ID" -screen "${GEOMETRY%x*}" -auth "$XAUTH" -ac -resizeable \
        -title "PlatynUI isolated $DISPLAY_ID" \
        >/tmp/platynui_bootstrap_xephyr.log 2>&1 &
    PIDS+=("$!")
    ;;
  vnc)
    command -v x11vnc >/dev/null || { echo "x11vnc required for vnc mode" >&2; exit 1; }
    start_xvfb
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 1;;
esac

# Wait for the display.
for _ in $(seq 1 20); do
  if XAUTHORITY="$XAUTH" DISPLAY="$DISPLAY_ID" xdpyinfo >/dev/null 2>&1; then break; fi
  sleep 0.5
done
if ! XAUTHORITY="$XAUTH" DISPLAY="$DISPLAY_ID" xdpyinfo >/dev/null 2>&1; then
  echo "isolated display $DISPLAY_ID did not become ready" >&2
  exit 1
fi

if [[ "$MODE" == "vnc" ]]; then
  x11vnc -display "$DISPLAY_ID" -auth "$XAUTH" -localhost -rfbport "$VNC_PORT" \
      -nopw -forever -quiet >/tmp/platynui_bootstrap_vnc.log 2>&1 &
  PIDS+=("$!")
  echo "VNC on localhost:$VNC_PORT (attach a viewer to observe)"
fi

# Start a minimal EWMH window manager inside the nested display (visible mode).
# PlatynUI's window activation (WindowSurface.activate / bring_to_front /
# window --list) is implemented via EWMH (_NET_ACTIVE_WINDOW,
# _NET_CLIENT_LIST) and needs a WM on the display; without one, rf-mcp's
# focus verification degrades to a focus-unverifiable warning. The isolation
# marker (not EWMH absence) is what classifies the display 'isolated', so a
# WM inside the nested display does not weaken the safety guard.
# change: platynui-visible-safe-targeting (task 3.5).
if [[ "$MODE" == "visible" ]]; then
  if command -v openbox >/dev/null; then
    XAUTHORITY="$XAUTH" DISPLAY="$DISPLAY_ID" openbox \
        >/tmp/platynui_bootstrap_wm.log 2>&1 &
    PIDS+=("$!")
    echo "EWMH WM (openbox) started on $DISPLAY_ID"
  else
    echo "NOTE: no EWMH WM available (openbox not found) — PlatynUI window" >&2
    echo "activation will be unavailable on $DISPLAY_ID; focus verification" >&2
    echo "degrades to a warning. Install openbox for verified focus." >&2
  fi
fi

# The confined environment (the documented recipe + the isolation marker).
export DISPLAY="$DISPLAY_ID"
export XAUTHORITY="$XAUTH"
export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11
export GSK_RENDERER=cairo
unset WAYLAND_DISPLAY || true
export ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY="$DISPLAY_ID"
# Ownership corroboration (change: desktop-isolation-marker-hardening): record
# the PID of the X server WE launched so the safety guard can verify the marker
# against a live X server for this display instead of trusting it on assertion.
export ROBOTMCP_PLATYNUI_ISOLATED_XPID="${PIDS[0]}"

if [[ $# -eq 0 ]]; then
  cat <<EOF
# PlatynUI isolated session ready ($MODE) on $DISPLAY_ID
export DISPLAY=$DISPLAY_ID
export XAUTHORITY=$XAUTH
export XDG_SESSION_TYPE=x11 GDK_BACKEND=x11 GSK_RENDERER=cairo
unset WAYLAND_DISPLAY
export ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=$DISPLAY_ID
export ROBOTMCP_PLATYNUI_ISOLATED_XPID=${PIDS[0]}
# Ctrl-C to tear down.
EOF
  while true; do sleep 3600; done
else
  "$@"
fi
