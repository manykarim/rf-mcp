#!/usr/bin/env bash
# Run the robotframework-agenteval harness against rf-mcp
# (change: adopt-agenteval-harness).
#
# ISOLATION: agenteval is installed into an EPHEMERAL uv env (--no-project
# --with-requirements). rf-mcp is spawned as a SUBPROCESS by the suites, running
# from its OWN .venv. agenteval's pinned deps never enter rf-mcp's environment.
#
# Usage (positional args are passed straight to `robot`; the target is this dir):
#   tests/agenteval/run.sh                                 # run the whole harness
#   tests/agenteval/run.sh --suite 'Deterministic Mcp Surface'   # one suite
#   tests/agenteval/run.sh --test 'Rf-mcp*'                # select tests
#
# The agentic suite skips itself unless AGENTEVAL_API_KEY (+ BASE_URL/MODEL) is set.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# The suites spawn .venv/bin/robotmcp; make sure rf-mcp's own env exists.
if [ ! -x .venv/bin/robotmcp ]; then
    echo "rf-mcp .venv not found - running 'uv sync' to build it..." >&2
    uv sync --group dev >/dev/null
fi

# Any args are robot options; the suite target is always this directory.
exec uv run --no-project --with-requirements "$HERE/requirements.txt" \
    robot --outputdir "$HERE/results" "$@" "$HERE"
