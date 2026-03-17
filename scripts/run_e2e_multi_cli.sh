#!/usr/bin/env bash
# Run BDD DemoShop e2e tests with copilot, codex, and claude CLIs
# Usage: ./scripts/run_e2e_multi_cli.sh [tool_filter]
# Examples:
#   ./scripts/run_e2e_multi_cli.sh          # Run all 3 tools
#   ./scripts/run_e2e_multi_cli.sh copilot  # Run only copilot
#   ./scripts/run_e2e_multi_cli.sh codex    # Run only codex
#   ./scripts/run_e2e_multi_cli.sh claude   # Run only claude

set -uo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR="tests/e2e/metrics/bdd_e2e_$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

FILTER="${1:-all}"

PROMPT='Use RobotMCP to test the demoshop. Execute step by step using the MCP tools.

Scenario: BDD Purchase Flow

1. Call manage_session to initialize a Browser session (headless=False)
2. Call get_locator_guidance for Browser library
3. Call execute_step: keyword="New Browser", arguments=["chromium", "headless=False"]
4. Call execute_step: keyword="New Page", arguments=["https://demoshop.makrocode.de/"]
5. Call get_session_state with include_reduced_dom=True to see the page
6. Find the "Sauce Labs Backpack" Add to Cart button in the DOM
7. Call execute_step to click it
8. Call get_session_state to verify cart badge shows 1
9. Find "Sauce Labs Bike Light" Add to Cart button
10. Call execute_step to click it
11. Call get_session_state to verify cart badge shows 2
12. Click the cart link/icon
13. Call get_session_state to verify both items in cart
14. Click Checkout button
15. Fill in First Name: Test, Last Name: User, Zip: 12345
16. Click Continue, then Finish
17. Call get_session_state to verify checkout success message
18. Call build_test_suite with bdd_style=True to generate the final .robot file
19. Show me the generated .robot file content

IMPORTANT: Always inspect the DOM with get_session_state(include_reduced_dom=True) before clicking.
Use real locators from the ARIA snapshot. Never guess locators.'

echo "=========================================="
echo "BDD DemoShop E2E - Multi-CLI Runner"
echo "Results dir: $RESULTS_DIR"
echo "Filter: $FILTER"
echo "=========================================="

# ── Copilot CLI ──────────────────────────────────────────────
run_copilot() {
    local logfile="$RESULTS_DIR/copilot.log"
    echo ""
    echo ">>> Running: GitHub Copilot CLI"
    echo "    Log: $logfile"
    echo "    Started: $(date +%H:%M:%S)"
    local start_time=$(date +%s)

    copilot \
        -p "$PROMPT" \
        --autopilot \
        --output-format json \
        > "$RESULTS_DIR/copilot.json" 2>"$logfile" || true

    local end_time=$(date +%s)
    echo "    Finished: $(date +%H:%M:%S) ($((end_time - start_time))s)"

    if [ -f "$RESULTS_DIR/copilot.json" ] && [ -s "$RESULTS_DIR/copilot.json" ]; then
        echo "    Output size: $(wc -c < "$RESULTS_DIR/copilot.json") bytes"
    else
        echo "    WARNING: No output captured"
    fi
}

# ── Codex CLI ────────────────────────────────────────────────
run_codex() {
    local logfile="$RESULTS_DIR/codex.log"
    echo ""
    echo ">>> Running: OpenAI Codex CLI"
    echo "    Log: $logfile"
    echo "    Started: $(date +%H:%M:%S)"
    local start_time=$(date +%s)

    codex exec \
        --full-auto \
        "$PROMPT" \
        > "$RESULTS_DIR/codex.txt" 2>"$logfile" || true

    local end_time=$(date +%s)
    echo "    Finished: $(date +%H:%M:%S) ($((end_time - start_time))s)"

    if [ -f "$RESULTS_DIR/codex.txt" ] && [ -s "$RESULTS_DIR/codex.txt" ]; then
        echo "    Output size: $(wc -c < "$RESULTS_DIR/codex.txt") bytes"
    else
        echo "    WARNING: No output captured"
    fi
}

# ── Claude CLI ───────────────────────────────────────────────
run_claude() {
    local logfile="$RESULTS_DIR/claude.log"
    echo ""
    echo ">>> Running: Claude Code CLI"
    echo "    Log: $logfile"
    echo "    Started: $(date +%H:%M:%S)"
    local start_time=$(date +%s)

    claude -p "$PROMPT" \
        --output-format json \
        --allowedTools "mcp__robotmcp__*" \
        --max-turns 50 \
        > "$RESULTS_DIR/claude.json" 2>"$logfile" || true

    local end_time=$(date +%s)
    echo "    Finished: $(date +%H:%M:%S) ($((end_time - start_time))s)"

    if [ -f "$RESULTS_DIR/claude.json" ] && [ -s "$RESULTS_DIR/claude.json" ]; then
        echo "    Output size: $(wc -c < "$RESULTS_DIR/claude.json") bytes"
    else
        echo "    WARNING: No output captured"
    fi
}

# ── Run selected tools ───────────────────────────────────────
if [ "$FILTER" = "all" ] || [ "$FILTER" = "copilot" ]; then
    run_copilot
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "codex" ]; then
    run_codex
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "claude" ]; then
    run_claude
fi

echo ""
echo "=========================================="
echo "All runs complete. Results in: $RESULTS_DIR"
echo "=========================================="
