#!/usr/bin/env bash
set -euo pipefail

DISPLAY_ID=${DISPLAY_ID:-:99}
SCREEN_GEOMETRY=${SCREEN_GEOMETRY:-1280x1024x24}
SUITE_PATH=${SUITE_PATH:-tests/e2e/gnome_calculator_mcp_stepwise.robot}
RESULTS_DIR=${RESULTS_DIR:-results}

cleanup() {
    if [[ -n "${XVFB_PID:-}" ]] && kill -0 "$XVFB_PID" 2>/dev/null; then
        kill "$XVFB_PID" 2>/dev/null || true
        wait "$XVFB_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

if ! command -v Xvfb >/dev/null 2>&1; then
    echo "Xvfb is required but not installed." >&2
    exit 1
fi

if ! command -v xdpyinfo >/dev/null 2>&1; then
    echo "xdpyinfo is required but not installed." >&2
    exit 1
fi

Xvfb "$DISPLAY_ID" -screen 0 "$SCREEN_GEOMETRY" -ac >/tmp/rf_calc_xvfb_out.log 2>/tmp/rf_calc_xvfb_err.log &
XVFB_PID=$!

for _ in 1 2 3 4 5 6 7 8 9 10; do
    if DISPLAY="$DISPLAY_ID" xdpyinfo >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! DISPLAY="$DISPLAY_ID" xdpyinfo >/dev/null 2>&1; then
    echo "Isolated Xvfb display $DISPLAY_ID did not become ready." >&2
    exit 1
fi

export DISPLAY="$DISPLAY_ID"
export XDG_SESSION_TYPE=x11
export GDK_BACKEND=x11
export GSK_RENDERER=cairo
unset WAYLAND_DISPLAY || true

uv run robot -d "$RESULTS_DIR" "$SUITE_PATH"