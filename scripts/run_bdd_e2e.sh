#!/usr/bin/env bash
# Run BDD DemoShop e2e tests with multiple models via opencode
# Usage: ./scripts/run_bdd_e2e.sh [model_filter]
# Examples:
#   ./scripts/run_bdd_e2e.sh              # Run all 3 models
#   ./scripts/run_bdd_e2e.sh qwen         # Run only Qwen3 Coder
#   ./scripts/run_bdd_e2e.sh glm          # Run only GLM 4.5 Air
#   ./scripts/run_bdd_e2e.sh minimax      # Run only Minimax M2

set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR="tests/e2e/metrics/bdd_demoshop_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

FILTER="${1:-all}"

# Models to test
declare -A MODELS
MODELS[qwen]="openrouter/qwen/qwen3-coder"
MODELS[glm]="openrouter/z-ai/glm-4.5-air"
MODELS[minimax]="openrouter/minimax/minimax-m2"

PROMPT='Use RobotMCP to create a test suite and execute it step wise.
Use BDD style (Given/When/Then/And) for all keywords where it makes sense.
Use Browser Library with headless=False.

Scenario: Purchase products from the demoshop

- Open https://demoshop.makrocode.de/
- Add "Sauce Labs Backpack" to the cart
- Assert the cart badge shows 1 item
- Add "Sauce Labs Bike Light" to the cart
- Assert the cart badge shows 2 items
- Open the cart
- Assert both items are in the cart
- Proceed to checkout
- Fill in checkout information (first name: Test, last name: User, zip: 12345)
- Continue and finish the checkout
- Assert the checkout is complete with a success message

Execute step by step using execute_step.
Use get_session_state with include_reduced_dom=True to inspect the page DOM/ARIA before interacting with elements.
Build the final test suite afterwards using build_test_suite.

IMPORTANT:
- Use real locators from the DOM inspection, never guess locators
- Use BDD prefixes (Given/When/Then/And) in execute_step keyword names where natural
- After each page navigation or action, inspect the DOM again with get_session_state'

echo "=========================================="
echo "BDD DemoShop E2E Test Runner"
echo "Results dir: $RESULTS_DIR"
echo "=========================================="

run_model() {
    local name="$1"
    local model="$2"
    local logfile="$RESULTS_DIR/${name}.log"
    local jsonfile="$RESULTS_DIR/${name}.json"

    echo ""
    echo ">>> Running model: $name ($model)"
    echo "    Log: $logfile"
    echo "    Started: $(date +%H:%M:%S)"

    local start_time=$(date +%s)

    # Run opencode with the model and capture output
    opencode run \
        --model "$model" \
        --format json \
        --title "BDD DemoShop E2E - $name" \
        "$PROMPT" \
        > "$jsonfile" 2>"$logfile" || true

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "    Finished: $(date +%H:%M:%S) (${duration}s)"

    # Extract key metrics from JSON output
    if [ -f "$jsonfile" ] && [ -s "$jsonfile" ]; then
        local tool_calls=$(grep -c '"type":"tool_call"' "$jsonfile" 2>/dev/null || echo "0")
        local tool_results=$(grep -c '"type":"tool_result"' "$jsonfile" 2>/dev/null || echo "0")
        echo "    Tool calls: $tool_calls, Tool results: $tool_results"
    else
        echo "    WARNING: No JSON output captured"
    fi

    # Also save in default format for readability
    local readable_log="$RESULTS_DIR/${name}_readable.log"
    opencode run \
        --model "$model" \
        --continue \
        --format default \
        "Show me a summary of what was done, including: which tools were called, whether the test passed, and the final .robot file content" \
        > "$readable_log" 2>&1 || true

    echo "    Summary: $readable_log"
}

for name in qwen glm minimax; do
    if [ "$FILTER" = "all" ] || [ "$FILTER" = "$name" ]; then
        run_model "$name" "${MODELS[$name]}"
    fi
done

echo ""
echo "=========================================="
echo "All runs complete. Results in: $RESULTS_DIR"
echo "=========================================="

# Print summary
echo ""
echo "=== SUMMARY ==="
for name in qwen glm minimax; do
    if [ "$FILTER" = "all" ] || [ "$FILTER" = "$name" ]; then
        local_json="$RESULTS_DIR/${name}.json"
        if [ -f "$local_json" ] && [ -s "$local_json" ]; then
            tool_calls=$(grep -c '"type":"tool_call"' "$local_json" 2>/dev/null || echo "0")
            errors=$(grep -c '"error"' "$local_json" 2>/dev/null || echo "0")
            echo "  $name (${MODELS[$name]}): $tool_calls tool calls, $errors potential errors"
        else
            echo "  $name (${MODELS[$name]}): NO OUTPUT"
        fi
    fi
done
