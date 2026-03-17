# BDD & Embedded Argument Experiment Results

**Date**: 2026-03-17
**Robot Framework version**: (installed via rf-mcp dependencies)
**Working directory**: `/home/many/workspace/rf-mcp`
**Branch**: `feature/support_bdd_and_dd`

---

## Experiment 1: RF BDD Prefix Stripping

**Objective**: Verify that Robot Framework correctly strips Given/When/Then prefixes and matches keywords defined without the prefix.

**Test file**: `/tmp/rf_bdd_exp1.robot`

```robot
*** Settings ***
Library    BuiltIn

*** Test Cases ***
BDD Prefix Test
    Given the system is ready
    When the user logs in
    Then the dashboard is visible

*** Keywords ***
the system is ready
    Log    System ready

the user logs in
    Log    User logged in

the dashboard is visible
    Log    Dashboard visible
```

**Result**: PASS (1 test, 1 passed, 0 failed)

**Keyword execution trace** (from output.xml):
```
Given the system is ready  -> PASS  (matched "the system is ready")
When the user logs in      -> PASS  (matched "the user logs in")
Then the dashboard is visible -> PASS (matched "the dashboard is visible")
```

**Findings**:
- RF natively strips `Given`, `When`, `Then` prefixes before keyword lookup.
- Keywords are defined WITHOUT the prefix; the BDD prefix is purely syntactic sugar.
- This is a core RF behavior, not a library feature.

---

## Experiment 2: Embedded Arguments

**Objective**: Verify that embedded arguments with `${variable}` in keyword names work correctly, including with BDD prefixes.

**Test file**: `/tmp/rf_bdd_exp2.robot`

```robot
*** Settings ***
Library    BuiltIn

*** Test Cases ***
Embedded Arg Test
    Given the user has product "Drone" in the cart
    When the user adds product "Camera" to the cart
    Then the cart should contain 2 items

*** Keywords ***
the user has product "${product}" in the cart
    Log    Adding ${product} to cart initially

the user adds product "${product}" to the cart
    Log    Adding ${product} to cart

the cart should contain ${count} items
    Log    Cart has ${count} items
    Should Be Equal As Integers    ${count}    2
```

**Result**: PASS (1 test, 1 passed, 0 failed)

**Keyword execution trace** (from output.xml):
```
Given the user has product "Drone" in the cart   -> PASS (product=Drone)
When the user adds product "Camera" to the cart  -> PASS (product=Camera)
Then the cart should contain 2 items             -> PASS (count=2)
```

**Findings**:
- BDD prefixes (Given/When/Then) are stripped BEFORE embedded argument matching.
- Both quoted (`"Drone"`) and unquoted (`2`) values match embedded args.
- Embedded args work in any position within the keyword name.
- `${count}` without quotes captures the bare value `2`.

---

## Experiment 3: Multi-level Embedded Args

**Objective**: Test complex embedded argument patterns with multiple test cases reusing the same embedded keywords.

**Test file**: `/tmp/rf_bdd_exp3.robot`

```robot
*** Settings ***
Library    BuiltIn
Library    Collections

*** Test Cases ***
Shop Flow With Embedded Args
    Given the user opens the "DemoShop" homepage
    When the user adds "Sauce Labs Backpack" to the cart
    And the user adds "Sauce Labs Bike Light" to the cart
    Then the cart badge should show "2" items
    When the user fills in "Test" as first name
    And the user fills in "User" as last name
    And the user fills in "12345" as zip code
    Then the checkout should be complete

Data Driven Variant
    Given the user opens the "AltShop" homepage
    When the user adds "Widget A" to the cart
    And the user adds "Widget B" to the cart
    Then the cart badge should show "2" items
```

**Result**: PASS (2 tests, 2 passed, 0 failed)

**Key findings**:
- `And` prefix is also stripped (like Given/When/Then).
- Keywords with multiple embedded args work: `the user fills in "${value}" as ${field}` matches both `"Test" as first name` and `"12345" as zip code`.
- The `${field}` embedded arg (without quotes) matches multi-word values like `first name`, `last name`, `zip code`.
- Keyword reuse across test cases works perfectly -- `the user opens the "${shop}" homepage` matched both "DemoShop" and "AltShop".
- `But` is also a valid BDD prefix (not tested here but documented in RF).

---

## Experiment 4: _generate_bdd_keyword_name Heuristic

**Objective**: Test the current heuristic with a realistic 17-step demoshop purchase flow to evaluate quality of auto-generated BDD keyword names.

**Input**: 17 steps (New Browser, New Page, Click x8, Get Text x5, Fill Text x3)

### Generated BDD Test Case (what the heuristic produces)

```
Given the demoshop is open
When the user fills in the form
Then the cart shows the expected value
When the user clicks add sauce labs bike light to cart
Then the cart shows the expected value
When the user clicks cart
Then the expected results are verified
When the user fills in the form
Then the complete header shows the expected value
```

### Generated Keywords

| # | Name | Intent | Embedded | Steps |
|---|------|--------|----------|-------|
| 1 | `the demoshop is open` | given | No | New Browser + New Page |
| 2 | `the user fills in the form` | when | No | Click Products + Click Add Backpack |
| 3 | `the c ${value} shows the expected value` | then | Yes | Get Text ${value} == ${arg2} |
| 4 | `the user clicks ${value} cart` | when | Yes | Click ${value} |
| 5 | `the expected results are verified` | then | No | 2x Get Text (contains) |

### Quality Analysis

**Total keywords**: 5 (2 embedded, 3 non-embedded)

**Issues found**:

1. **WRONG GROUPING** -- Step 2 "the user fills in the form" groups `Click Products` + `Click Add Backpack to cart` together. These are NOT a form fill. The grouping heuristic treated consecutive `Click` + `Click` as a form fill group (multi-step `when` with mix of actions defaults to "fills in the form").

2. **MALFORMED EMBEDDED NAME** -- `the c ${value} shows the expected value` is generated from merging:
   - `the cart shows the expected value` (for Get Text on cart badge)
   - `the complete header shows the expected value` (for Get Text on .complete-header)

   The character-level `_build_embedded_name` finds common prefix "the c" (shortest common prefix of "the cart..." and "the complete..."), producing a nonsensical keyword name.

3. **INCORRECT EMBEDDED MERGE** -- `the user clicks ${value} cart` is generated from merging click keywords that differ only in their locator. The suffix "cart" comes from one of the names (`the user clicks cart`) bleeding into the common suffix detection. The other names like `the user clicks checkout` and `the user clicks finish` don't end in "cart".

4. **VAGUE NAME** -- `the expected results are verified` is a generic fallback for the multi-step `then` group (2 Get Text with `contains`). It loses all semantic meaning about what's being verified (cart contents).

5. **SECOND "the user fills in the form"** -- The Fill Text x3 + Click Continue + Click Finish group is also named "the user fills in the form" (deduplication reuses the first keyword, but semantically the second occurrence wraps a different flow -- filling checkout form + completing order).

6. **MISSING CONTEXT** -- Individual product additions (Click "Add Backpack", Click "Add Bike Light") are not distinguishable in the BDD output. A human would write these as `the user adds "${product}" to the cart`.

### Individual Name Generation (debugging)

| Keyword | Locator | Generated Name |
|---------|---------|---------------|
| Click | `role=link[name="Products"]` | `the user clicks products` |
| Click | `role=button[name="Add Sauce Labs Backpack to cart"]` | `the user clicks add sauce labs backpack to cart` |
| Click | `role=link[name="Cart"]` | `the user clicks cart` |
| Click | `text="Checkout"` | `the user clicks checkout` |
| Click | `text="Continue"` | `the user clicks continue` |
| Click | `text="Finish"` | `the user clicks finish` |
| Get Text | `role=link[name="Cart"]` | `the cart shows the expected value` |
| Get Text | `.cart-item:nth-child(1)` | `the cart item:nth child(1) shows the expected value` |
| Get Text | `.cart-item:nth-child(2)` | `the cart item:nth child(2) shows the expected value` |
| Get Text | `.complete-header` | `the complete header shows the expected value` |

**Key observation**: The `_humanize_locator` method strips CSS classes/IDs into space-separated words, producing names like "cart item:nth child(1)" which contain CSS syntax artifacts.

### Full .robot output

```robot
*** Settings ***
Library         Browser

*** Test Cases ***
DemoShop Purchase
    Given the demoshop is open
    When the user fills in the form
    Then the cart shows the expected value
    When the user clicks add sauce labs bike light to cart
    Then the cart shows the expected value
    When the user clicks cart
    Then the expected results are verified
    When the user fills in the form
    Then the complete header shows the expected value

*** Keywords ***
the demoshop is open
    New Browser    chromium    headless=False
    New Page    https://demoshop.makrocode.de/

the user fills in the form
    Click    role=link[name="Products"]
    Click    role=button[name="Add Sauce Labs Backpack to cart"]

the c ${value} shows the expected value
    Get Text    ${value}    ==    ${arg2}

the user clicks ${value} cart
    Click    ${value}

the expected results are verified
    Get Text    .cart-item:nth-child(1)    contains    Backpack
    Get Text    .cart-item:nth-child(2)    contains    Bike Light
```

---

## Experiment 5: Ideal BDD Output Validation

**Objective**: Verify that a hand-crafted ideal BDD .robot file for the same demoshop scenario parses correctly in Robot Framework.

**Test file**: `/tmp/rf_bdd_exp5.robot` -- a human-authored BDD suite with:
- 12 BDD steps in test case (Given/When/Then/And)
- 9 user keywords (3 embedded, 1 with [Arguments])
- `[Documentation]`, `Test Tags`

**Result**: PASS (1 test, 1 passed, 0 failed) via `--dryrun`

**Findings**:
- The ideal BDD structure parses and validates correctly in RF `--dryrun` mode.
- `[Arguments]` in keywords (e.g., `the user enters checkout details`) works alongside embedded args.
- Mixed embedded args (`${product}` in name) and regular args (`[Arguments] ${first_name}`) in different keywords within the same suite is valid.
- BDD prefixes work on keywords that have `[Arguments]`.

---

## Summary: Comparison of Heuristic vs. Ideal

| Aspect | Heuristic (Exp 4) | Ideal (Exp 5) |
|--------|-------------------|---------------|
| **Test case steps** | 9 BDD steps | 12 BDD steps |
| **Keywords** | 5 (2 embedded) | 9 (3 embedded, 1 with [Arguments]) |
| **Readability** | Poor -- vague names, wrong grouping | Excellent -- domain-specific language |
| **Semantic accuracy** | 2/5 keywords accurate | 9/9 keywords accurate |
| **Product parameterization** | Missing entirely | `the user adds "${product}" to the cart` |
| **Cart badge assertion** | Merged into malformed `the c ${value}...` | `the cart badge should show "${count}" items` |
| **Checkout form** | Grouped with navigation clicks | Separate keyword with [Arguments] |
| **Cart content verification** | Generic "expected results are verified" | `the cart should contain "${product}"` |

### Key Heuristic Weaknesses

1. **Step grouping is too coarse**: Consecutive same-intent steps are merged into one keyword even when they represent distinct logical actions (e.g., "navigate to products" + "add item to cart" are both `Click` -> `when`).

2. **`_build_embedded_name` character-level prefix/suffix is fragile**: Produces malformed names like `the c ${value} shows the expected value` when the varying part is mid-word.

3. **No domain awareness**: The heuristic has no concept of "adding to cart" vs. "navigating". It only knows keyword types (Click, Fill Text, Get Text).

4. **Multi-step `then` groups lose specificity**: `the expected results are verified` is a catch-all that loses information about what is being verified.

5. **CSS locator artifacts leak into names**: `.cart-item:nth-child(1)` becomes "cart item:nth child(1)" rather than a meaningful description.

6. **No `[Arguments]` generation for multi-param keywords**: The heuristic only uses embedded args, missing the opportunity to use `[Arguments]` for keywords that take multiple values (like checkout form fill).

### Recommendations for Improvement

1. **Word-level prefix/suffix matching** in `_build_embedded_name` instead of character-level.
2. **Semantic step grouping** that considers the locator target, not just the keyword type (Click on a product button vs. Click on a navigation link).
3. **Cart/assertion pattern recognition**: Detect `Get Text ... == <number>` patterns as "count assertions" and generate `should show "${count}"` names.
4. **CSS sanitization**: Strip nth-child, pseudo-selectors, and other CSS syntax from humanized locator names.
5. **Multi-value keyword detection**: When 3+ consecutive `Fill Text` steps target related fields, generate a single keyword with `[Arguments]` instead of one embedded keyword.
