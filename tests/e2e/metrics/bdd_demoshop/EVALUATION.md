# BDD DemoShop E2E Evaluation — ADR-019 Validation

**Date**: 2026-03-07
**Scenario**: Purchase products from demoshop.makrocode.de with BDD-style keywords
**Models**: Qwen3 Coder, GLM 4.5 Air, Minimax M2 (via OpenRouter)
**Tool**: opencode 1.2.15 with robotmcp MCP server

## Summary Table

| Metric | Qwen3 Coder | GLM 4.5 Air | Minimax M2 |
|--------|------------|-------------|------------|
| **Checkout Complete** | Yes (ORD-2E8CC0C3) | No | Yes (ORD-63EEB725) |
| **execute_step calls** | 82 | 1 | 20 |
| **Success rate** | 90% (74/82) | 0% (0/1) | 55% (11/20) |
| **Total tool calls** | 243 | 18 | 43 |
| **robotmcp tool calls** | 98 | 17 | 32 |
| **BDD-prefixed keywords** | 4 | 0 | 0 |
| **Used get_session_state** | 5x | 0x | 7x |
| **Built test suite** | Yes | No | Yes |
| **Cost** | $0.15 | $0.11 | $1.22 |
| **Input tokens** | 182K | 63K | 2.2M |
| **Output tokens** | 75K | 38K | 6K |
| **Steps (LLM rounds)** | 243 | 19 | 44 |

## Model Analysis

### Qwen3 Coder — Best Overall
- **Completed full checkout** with order confirmation
- **Used BDD prefixes**: "Given Navigate to demoshop website", "When Add first product to cart", "And Add second product to cart", "Then Assert cart shows 2 items"
- **BDD keywords failed** (expected — these are custom keywords not registered in RF) but the model correctly fell back to standard Browser Library keywords
- **Thorough DOM exploration**: Inspected page structure, found products by CSS class, used search input, enumerated buttons
- **Wrote embedded argument keywords**: `Add "${product}" to cart`, `Verify cart shows ${count}` (demonstrates ADR-019 embedded args understanding)
- **Issue**: Got stuck in a file-write loop after completing the workflow (129 write tool calls repeating same content). The robotmcp workflow itself completed successfully
- **Cost-efficient**: $0.15 for 82 execute_step calls

### GLM 4.5 Air — Failed Early
- **Failed at step 1**: Tried `Go To` without initializing a browser first
- **Got stuck in reasoning loop**: Repeatedly searched for SeleniumLibrary keywords despite prompt specifying Browser Library
- **Reasoning loop visible**: The model's internal reasoning repeated "I need to find the SeleniumLibrary 'Open Browser' keyword specifically" 21 times consecutively
- **Never recovered**: Spent 18 tool calls (7 manage_session, 6 find_keywords, 2 analyze_scenario) without making progress
- **Root cause**: GLM 4.5 Air fixated on SeleniumLibrary instead of following the Browser Library instruction

### Minimax M2 — Clean & Efficient
- **Completed full checkout** with order confirmation
- **Most efficient workflow**: Only 20 execute_step calls to complete the entire scenario
- **Adapted to actual shop content**: Correctly identified demoshop as "Flowline Supply" (not SauceLabs) and used actual product names (Cascade Water Bottle, Echo Conference Speaker)
- **Used ARIA/role locators**: `role=link[name="Products"]`, `role=button[name="Add Cascade Water Bottle to cart"]`, `role=textbox[name="Email"]` — clean, semantic locators
- **Higher failure rate (55%)** but recovered well from errors (tried multiple locator strategies)
- **Generated clean .robot file** that passes dry-run validation
- **No BDD prefixes used** — keywords were standard Browser Library names
- **Most expensive**: $1.22 due to massive 2.2M input tokens (likely prompt caching overhead on OpenRouter)

## ADR-019 BDD Feature Validation

### BDD Prefix Stripping (Phase 1-2)
- **Qwen3 Coder** attempted 4 BDD-prefixed keywords: `Given Navigate to demoshop website`, `When Add first product to cart`, `And Add second product to cart`, `Then Assert cart shows 2 items`
- These correctly triggered the BDD prefix stripping in `execute_step`, but failed because the stripped keyword names were custom (not registered in RF's namespace)
- **Validated**: The server correctly logged BDD prefix stripping and returned `bdd_prefix` info in responses
- **Insight**: BDD prefixes are most useful with user-defined keywords (Resource files), not with library keywords like Click/Fill Text

### Embedded Arguments (Phase 3)
- **Qwen3 Coder** wrote embedded argument keywords in its .robot file: `Add "${product}" to cart`, `Verify cart shows ${count}`
- These demonstrate understanding of the RF embedded argument pattern
- The execute_step calls themselves used standard keywords (embedded matching happens at suite-run level, not step-by-step execution)

### Data-Driven / Template Support (Phase 4)
- Not directly tested in this scenario (would need explicit `template` parameter in manage_session)
- The scenario structure (multiple products, different checkout data) is suitable for data-driven conversion

## Generated .robot Files

### Minimax M2 (Clean, Validated)
```robot
*** Test Cases ***
Purchase Products From Demo Shop
    New Browser    chromium    headless=False
    New Page    https://demoshop.makrocode.de/
    Click    role=link[name="Products"]
    Click    role=button[name="Add Cascade Water Bottle to cart"] >> nth=0
    Click    role=button[name="Add Echo Conference Speaker to cart"]
    Click    text=Cart 2
    Click    role=link[name="Proceed to checkout"]
    Fill Text    role=textbox[name="Email"]    test@example.com
    Fill Text    role=textbox[name="Full name"]    Test User
    Fill Text    role=textbox[name="Address"]    12345
    Click    role=button[name="Place order"]
    [Teardown]    Close Browser
```

### Qwen3 Coder (BDD Keywords, Not Validated as Suite)
```robot
*** Keywords ***
Add "${product}" to cart
    Fill Text    input[type="search"]    ${product}
    Press Keys    input[type="search"]    Enter
    Click    button:has-text("Add to Cart") >> nth=0

Verify cart shows ${count}
    ${cart_text}=    Get Text    [class*="cart"]
    Should Contain    ${cart_text}    ${count}
```

## Recommendations

1. **Model Selection**: Minimax M2 produced the cleanest output (fewest steps, best locators, clean .robot file) but at highest cost. Qwen3 Coder was thorough and cost-effective. GLM 4.5 Air is not suitable for this task.

2. **BDD Optimization**: The prompt should clarify that BDD prefixes work with user-defined keywords, not raw library keywords. Future prompts should instruct models to create BDD-style keyword abstractions first, then call them with Given/When/Then prefixes.

3. **Locator Strategy**: Minimax M2's ARIA role locators (`role=button[name="..."]`) are more robust than Qwen3 Coder's CSS selectors (`button:has-text("...")`). The `get_session_state` with `include_reduced_dom=True` effectively guided both models to use real DOM locators.

4. **Cost Control**: Minimax M2's 2.2M input tokens suggest inefficient context handling. Qwen3 Coder achieved better results at 1/8th the cost.
