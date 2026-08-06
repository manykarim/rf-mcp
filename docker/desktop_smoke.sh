#!/bin/bash
# Deterministic desktop smoke wrapper (change: desktop-docker-agent-harness).
# Runs the steering + read-back driver inside the project venv. Meant to be the
# default CMD behind entrypoint.sh (which has already brought the desktop up).
set -uo pipefail

echo "[smoke] platynui-cli providers:"
platynui-cli list-providers 2>&1 | sed 's/^/[smoke]   /' || echo "[smoke]   (platynui-cli unavailable)"

echo "[smoke] running steering + read-back driver…"
uv run --no-sync python /app/docker/desktop_smoke_driver.py
code=$?
echo "[smoke] driver exit=${code}  (0 = PASS: provider up, read-back 42, screenshot written)"
exit "${code}"
