#!/usr/bin/env bash
#
# Setup script for the LibreOffice Writer PlatynUI validation suite
# (tests/lo_validation/libreoffice_writer.robot).
#
# Provisions:
#   - An isolated Xvfb display on :99 (no on-host window, not headless-
#     visible; headless CI mode). For interactive (visible) mode, replace
#     with `scripts/platynui_desktop_bootstrap.sh --mode visible -- :99`.
#   - A clean LibreOffice user profile (skips first-run / tip-of-the-day
#     dialogs).
#   - A wrapper that launches robotmcp with the isolation marker so the
#     active-desktop safety guard classifies the display 'isolated'.
#
# Usage:
#   scripts/lo_validation_setup.sh           # provision + print run command
#   scripts/lo_validation_setup.sh run       # provision + execute the suite
#
# Tear down:
#   scripts/lo_validation_setup.sh teardown
#
set -euo pipefail

ISOLATED_DISPLAY="${ISOLATED_DISPLAY:-:99}"
LO_PROFILE="${LO_PROFILE:-/tmp/lo_profile99}"
TEST_DIR="${TEST_DIR:-/tmp/kilo/lo-test}"
GEOMETRY="1280x1024x24"
LOG_DIR="${LOG_DIR:-/tmp}"
SUITE_PATH="${SUITE_PATH:-tests/lo_validation/libreoffice_writer.robot}"

log() { printf '[lo-setup] %s\n' "$*" >&2; }

start_xvfb() {
  if pgrep -f "Xvfb ${ISOLATED_DISPLAY} " >/dev/null 2>&1; then
    log "Xvfb ${ISOLATED_DISPLAY} already running"
    return
  fi
  log "Starting Xvfb ${ISOLATED_DISPLAY} (${GEOMETRY})"
  nohup Xvfb "${ISOLATED_DISPLAY}" -screen 0 "${GEOMETRY}" -ac \
      >"${LOG_DIR}/platynui_bootstrap_xvfb.log" 2>&1 &
  disown
  for _ in $(seq 1 20); do
    if DISPLAY="${ISOLATED_DISPLAY}" xdpyinfo >/dev/null 2>&1; then
      log "Xvfb ${ISOLATED_DISPLAY} is ready"
      return
    fi
    sleep 0.3
  done
  log "ERROR: Xvfb ${ISOLATED_DISPLAY} did not become ready" >&2
  exit 1
}

prepare_lo_profile() {
  if [[ -f "${LO_PROFILE}/user/registrymodifications.xcu" ]]; then
    log "LibreOffice profile ${LO_PROFILE} already prepared"
    return
  fi
  log "Preparing clean LibreOffice profile at ${LO_PROFILE}"
  rm -rf "${LO_PROFILE}"
  mkdir -p "${LO_PROFILE}/user"
  cat > "${LO_PROFILE}/user/registrymodifications.xcu" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>
 <item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" oor:op="fuse"><value>false</value></prop></item>
</oor:items>
EOF
}

prepare_test_dir() {
  mkdir -p "${TEST_DIR}"
}

teardown() {
  log "Tearing down Xvfb ${ISOLATED_DISPLAY} and any leftover soffice"
  pkill -9 -f "soffice.bin" 2>/dev/null || true
  pkill -9 -f "Xvfb ${ISOLATED_DISPLAY} " 2>/dev/null || true
}

run_suite() {
  log "Running ${SUITE_PATH} with isolation env (DISPLAY=${ISOLATED_DISPLAY})"
  DISPLAY="${ISOLATED_DISPLAY}" \
  ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY="${ISOLATED_DISPLAY}" \
  ROBOTMCP_PLATYNUI_SAFETY_GUARD=warn \
  XDG_SESSION_TYPE=x11 \
  GDK_BACKEND=x11 \
  GSK_RENDERER=cairo \
  SAL_USE_VCLPLUGIN=gtk3 \
  GTK_A11Y=1 \
  uv run --project . robot \
      --variable "LO_PATH:/usr/bin/soffice" \
      --variable "LO_PROFILE:${LO_PROFILE}" \
      --variable "TEST_DIR:${TEST_DIR}" \
      --variable "TEST_FILE:${TEST_DIR}/my_document.fodt" \
      --variable "ISOLATED_DISPLAY:${ISOLATED_DISPLAY}" \
      "${SUITE_PATH}"
}

case "${1:-}" in
  teardown) teardown ;;
  run)
    start_xvfb
    prepare_lo_profile
    prepare_test_dir
    run_suite
    ;;
  ""|setup)
    start_xvfb
    prepare_lo_profile
    prepare_test_dir
    log "Ready. To run the suite:"
    log "  ${0} run"
    log "Or manually:"
    log "  export DISPLAY=${ISOLATED_DISPLAY}"
    log "  export ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=${ISOLATED_DISPLAY}"
    log "  uv run --project . robot ${SUITE_PATH}"
    ;;
  *) log "Usage: $0 [setup|run|teardown]" >&2; exit 2 ;;
esac
