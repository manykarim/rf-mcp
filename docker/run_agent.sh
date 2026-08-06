#!/bin/bash
# Agent rung (key-gated) — change: desktop-docker-agent-harness.
# Drives the in-container desktop via a MiniMax-backed opencode agent talking
# to robotmcp over stdio. Meant to run behind entrypoint.sh (desktop already up).
#
#   docker run --rm -e MINIMAX_API_KEY=sk-... [-e MINIMAX_BASE_URL=...] \
#       -v "$PWD/artifacts:/artifacts" robotmcp-desktop /app/docker/run_agent.sh
#
# Without a key, this skips gracefully (exit 0) — the deterministic smoke is the
# real gate.
set -uo pipefail

: "${MINIMAX_BASE_URL:=https://api.minimaxi.com/v1}"
# NOTE: keep this default free of apostrophes/backticks — it is a shell default
# in a double-quoted ${VAR:-...} expansion (a stray ' broke an earlier run).
DEFAULT_PROMPT="Open the calculator application on this desktop, compute 7 times 6, then CONFIRM the result two ways: (1) take a screenshot to /artifacts, and (2) read the calculator result display back via a PlatynUI keyword and state the value. Use only robotmcp tools; scope desktop locators to the application, e.g. /app:*[@Name=X]//control:..."
PROMPT="${AGENT_PROMPT:-$DEFAULT_PROMPT}"

if [ -z "${MINIMAX_API_KEY:-}" ]; then
    echo "[agent] MINIMAX_API_KEY not set — skipping the agent rung (the deterministic"
    echo "[agent] smoke is the acceptance gate). To run: docker run -e MINIMAX_API_KEY=... "
    echo "[agent] -e MINIMAX_BASE_URL=${MINIMAX_BASE_URL} ... /app/docker/run_agent.sh"
    exit 0
fi

export MINIMAX_BASE_URL

# opencode on demand (kept out of the image build so a broken install never
# blocks the deterministic smoke). Use the OFFICIAL installer — it drops a
# compiled binary in ~/.opencode/bin (user-writable), unlike `npm i -g` which
# EACCES'es on /usr/local for the non-root appuser.
export PATH="${HOME}/.opencode/bin:${HOME}/.local/bin:${PATH}"
if ! command -v opencode >/dev/null 2>&1; then
    echo "[agent] installing opencode (official installer)…"
    curl -fsSL https://opencode.ai/install | bash >/tmp/opencode-install.log 2>&1 || {
        echo "[agent] opencode install failed:"; tail -15 /tmp/opencode-install.log; exit 0; }
    export PATH="${HOME}/.opencode/bin:${HOME}/.local/bin:${PATH}"
fi
echo "[agent] opencode $(opencode --version 2>/dev/null || echo '?')"

mkdir -p "${HOME}/.config/opencode"
cp /app/docker/opencode.minimax.json "${HOME}/.config/opencode/opencode.json"

echo "[agent] running MiniMax agent against robotmcp…"
cd /app
opencode run "${PROMPT}" 2>&1 | tee /artifacts/agent_transcript.txt
echo "[agent] transcript -> /artifacts/agent_transcript.txt ; screenshots -> /artifacts"
