# Proposal: BDD Scenario Quality Improvement

**Date**: 2026-03-17
**Status**: PROPOSAL (not yet implemented)
**Author**: Research synthesis from 3 parallel analyses + 5 RF experiments

---

## Executive Summary

The current BDD heuristic in `build_test_suite(bdd_style=True)` produces functional but low-quality BDD output. A realistic 17-step demoshop purchase flow generates:

- **Malformed keyword names**: `the c ${value} shows the expected value`
- **Wrong groupings**: "Click Products" + "Click Add Backpack" merged as "the user fills in the form"
- **Generic fallbacks**: `the expected results are verified` losing all domain meaning
- **CSS artifacts leaking**: `.cart-item:nth-child(1)` → "cart item:nth child(1)"

This proposal outlines 8 improvements organized in 3 phases, validated by RF experiments confirming that the target output structure is valid Robot Framework syntax.

---

## Problem Evidence

### Current Output (from Experiment 4 — realistic 17-step demoshop flow)

```robot
*** Test Cases ***
DemoShop Purchase
    Given the demoshop is open
    When the user fills in the form        ← WRONG: this is Click Products + Click Add Backpack
    Then the cart shows the expected value
    When the user clicks add sauce labs bike light to cart
    Then the cart shows the expected value
    When the user clicks cart
    Then the expected results are verified  ← VAGUE: losing cart content verification detail
    When the user fills in the form         ← REUSED: different flow (checkout) same name
    Then the complete header shows the expected value

*** Keywords ***
the c ${value} shows the expected value    ← MALFORMED: character-level prefix merge
    Get Text    ${value}    ==    ${arg2}

the user clicks ${value} cart              ← WRONG SUFFIX: "cart" from one name only
    Click    ${value}
```

### Ideal Output (validated in Experiment 5 — passes RF `--dryrun`)

```robot
*** Test Cases ***
Purchase Products From DemoShop
    Given the user opens the demoshop
    When the user adds "Sauce Labs Backpack" to the cart
    Then the cart badge should show "1" items
    When the user adds "Sauce Labs Bike Light" to the cart
    Then the cart badge should show "2" items
    When the user opens the cart
    Then the cart should contain "Sauce Labs Backpack"
    And the cart should contain "Sauce Labs Bike Light"
    When the user proceeds to checkout
    And the user enters checkout details    Test    User    12345
    And the user completes the order
    Then the order confirmation should be displayed

*** Keywords ***
the user opens the demoshop
    New Browser    chromium    headless=False
    New Page    https://demoshop.makrocode.de/

the user adds "${product}" to the cart
    Click    role=button[name="Add ${product} to cart"]

the cart badge should show "${count}" items
    Get Text    css=.shopping_cart_badge    ==    ${count}

the user enters checkout details
    [Arguments]    ${first_name}    ${last_name}    ${zip}
    Fill Text    id=first-name    ${first_name}
    Fill Text    id=last-name    ${last_name}
    Fill Text    id=postal-code    ${zip}
```

---

## Root Cause Analysis

### 1. Single-Signal Grouping (Critical)

`_group_by_heuristics` uses only intent transitions (given→when→then) to split groups. A sequence of 8 consecutive Clicks (all classified as "when") becomes one monolithic keyword. Real flows have semantic phases (browse → cart → checkout → confirm) that are invisible to the intent classifier.

### 2. Character-Level Embedded Name Building (Bug)

`_build_embedded_name` finds common prefix/suffix by character, not by word. Merging "the cart shows..." and "the complete header shows..." finds prefix "the c" (stops at character divergence), producing `the c ${value} shows the expected value`.

### 3. No Instrumentation Step Filtering

LLM agents call `Get Text body` (read entire DOM) between actions for navigation decisions. These are not test assertions but get classified as "then", causing rapid when/then/when/then oscillation and dozens of tiny groups with generic names.

### 4. Multi-Step Name Generation Falls Back to Generic

For multi-step groups, `_generate_bdd_keyword_name` checks if steps are all Clicks or all Fills, but falls back to "the user fills in the form" or "the user performs the action" for mixed groups, losing all domain meaning.

### 5. No Locator Semantic Analysis

ARIA locators contain rich information: `role=button[name="Add Sauce Labs Backpack to cart"]` contains the product name and the action (add to cart). This is used for humanization but never fed back into grouping or embedded arg detection.

### 6. Embedded Args Detected After Reference Assignment (Correctness Bug)

`_detect_embedded_args` runs after test case steps already reference keywords by name. When keywords are merged, the names change but the test case references are NOT updated, producing mismatched names in the output.

---

## Proposed Improvements

### Phase 1: Quick Wins (Low Risk, High Impact)

#### 1.1 Word-Level Embedded Name Building

**Problem**: Character-level prefix/suffix produces `the c ${value} shows...`
**Fix**: Split names into words, find common word prefix/suffix, replace differing words with `${arg}`.

```python
# Current (character-level):
# "the cart shows..." + "the complete header shows..."
# → prefix "the c", suffix "shows the expected value"
# → "the c ${value} shows the expected value"

# Proposed (word-level):
# words: ["the", "cart", "shows", ...] vs ["the", "complete", "header", "shows", ...]
# → prefix words: ["the"]
# → suffix words: ["shows", "the", "expected", "value"]
# → "the ${value} shows the expected value"
```

**Experiment validation**: Experiment 2 confirmed RF embedded args work with word-boundary `${var}` in any position.

#### 1.2 Instrumentation Step Filtering

**Problem**: `Get Text body`, bare `Get Url`, `Get Title` without assertion operators pollute BDD output.
**Fix**: Before BDD transformation, filter steps where:
- Keyword is `Get Text` and first argument is `body` (whole-page read)
- Keyword is `Get Url`/`Get Title` without `==`/`should`/`contains` operator
- Keyword is `Get Property` without assertion operator

```python
_INSTRUMENTATION_PATTERNS = [
    lambda s: s.keyword.lower() == "get text" and s.arguments and s.arguments[0].lower() == "body"
              and len(s.arguments) == 1,  # Get Text body (no assertion)
    lambda s: s.keyword.lower() in ("get url", "get title") and len(s.arguments) <= 1,
    lambda s: s.keyword.lower() == "get property" and len(s.arguments) <= 2,  # No assertion
]
```

#### 1.3 Fix "the the" Double Article Bug

**Problem**: `_generate_bdd_keyword_name` produces `"the {target}"` where target can already start with "the" (from `_humanize_locator` fallback "the element"), giving "the the element".
**Fix**: Strip leading "the " from target before interpolation.

```python
target = self._humanize_locator(args[0]) if args else "the element"
if target.startswith("the "):
    target = target[4:]
return f"the {target} shows the expected value"
```

#### 1.4 CSS Locator Sanitization

**Problem**: `.cart-item:nth-child(1)` becomes "cart item:nth child(1)" with CSS syntax artifacts.
**Fix**: Strip CSS pseudo-selectors and structural selectors before humanization.

```python
# Add to _humanize_locator:
if loc.startswith("."):
    name = loc[1:]
    # Strip CSS pseudo-selectors
    name = re.sub(r':[\w-]+\([^)]*\)', '', name)  # :nth-child(1), :first-child, etc.
    name = re.sub(r':[\w-]+', '', name)             # :hover, :focus, etc.
    return name.replace("-", " ").replace("_", " ").strip()
```

**Estimated effort**: 4-6 hours for all Phase 1 items.

---

### Phase 2: Semantic Grouping (Medium Risk, High Impact)

#### 2.1 Phase-Aware Step Grouping

**Problem**: All Clicks are "when" regardless of semantic phase.
**Fix**: Detect semantic phases from locator content using indicator words.

```python
_PHASE_INDICATORS = {
    "cart":     {"cart", "basket", "bag", "add to cart", "remove from cart", "add .* to cart"},
    "checkout": {"checkout", "proceed", "continue", "shipping", "payment", "billing"},
    "order":    {"place order", "finish", "complete order", "submit order", "confirm"},
    "login":    {"login", "sign in", "log in", "username", "password"},
    "search":   {"search", "filter", "sort by"},
    "navigate": {"products", "home", "categories", "menu"},
}

def _detect_phase(self, step: TestCaseStep) -> Optional[str]:
    """Detect semantic phase from step's locator content."""
    if not step.arguments:
        return None
    humanized = self._humanize_locator(step.arguments[0]).lower()
    for phase, indicators in self._PHASE_INDICATORS.items():
        for indicator in indicators:
            if re.search(indicator, humanized):
                return phase
    return None
```

Phase changes force group boundaries even within the same intent (when→when transition when phase changes from "navigate" to "cart").

#### 2.2 Max Group Size Constraint

**Problem**: 30-step "when" runs produce unreadable keywords.
**Fix**: Split groups exceeding 5-7 steps at natural boundaries (action type change: fill→click, or locator domain change).

```python
def _enforce_group_size_limits(self, groups, max_size=7):
    result = []
    for group in groups:
        if len(group.steps) <= max_size:
            result.append(group)
            continue
        # Split at action-type transitions
        sub_groups = self._split_at_action_boundaries(group, max_size)
        result.extend(sub_groups)
    return result
```

#### 2.3 Multi-Step Name Improvement with Locator Analysis

**Problem**: Multi-step "when" groups default to generic names.
**Fix**: Use the last significant action's humanized locator as the name anchor.

```python
# Multi-step "when" group naming:
# 1. If all steps are Fill → "the user fills in the {form_context}"
# 2. If all steps are Click → use the LAST click's target (often the submit action)
# 3. If mixed → use the phase name: "the user completes the {phase} step"

# Multi-step "then" group naming:
# 1. Count assertions vs. bare getters
# 2. Use the first assertion's target: "the {target} is verified"
# 3. If has should-keyword: extract the condition
```

**Estimated effort**: 8-12 hours for all Phase 2 items.

---

### Phase 3: Embedded Argument Intelligence (Higher Risk, Transformative Impact)

#### 3.1 Two-Pass Embedded Arg Architecture

**Problem**: Embedded args are detected after test case references are already set.
**Fix**: Restructure `_transform_to_bdd_style` into 3 passes:

```
Pass 1: Group steps → create preliminary BddKeyword objects
Pass 2: Detect & merge embedded args → update keyword names
Pass 3: Assign test case references using final merged names
```

This ensures the test case body always matches the Keywords section.

#### 3.2 Locator-Aware Embedded Arg Detection

**Problem**: The current detector uses structural comparison (same keyword + same arg count) to find merge candidates. This misses semantic opportunities where the locator itself contains the varying value.

**Example**: `Click role=button[name="Add Sauce Labs Backpack to cart"]` and `Click role=button[name="Add Sauce Labs Bike Light to cart"]` — the locator itself shows the `${product}` pattern.

**Fix**: Before structural merging, scan for locators containing user-facing text that varies between similar steps. Extract the varying text as an embedded argument:

```python
def _detect_locator_embedded_args(self, keywords):
    """Pre-merge pass: detect ${var} patterns in locator text."""
    for kw in keywords:
        if len(kw.steps) == 1:
            step = kw.steps[0]
            if step.arguments:
                loc = step.arguments[0]
                # Check if locator contains a name/text with quotes
                match = re.search(r'name="([^"]+)"', loc)
                if match:
                    text = match.group(1)
                    # Look for known patterns: "Add X to cart", "Select X"
                    for pattern, arg_name in self._LOCATOR_PATTERNS:
                        m = re.match(pattern, text)
                        if m:
                            kw.embedded_args[arg_name] = m.group(1)
                            kw.is_embedded = True
                            # Rewrite locator with ${arg}
                            new_loc = loc.replace(text, re.sub(pattern, r'${' + arg_name + '}', text))
                            step.arguments[0] = new_loc
                            break
    return keywords
```

#### 3.3 [Arguments] Keyword Generation for Multi-Value Steps

**Problem**: 3+ consecutive Fill Text targeting related fields produce 3 separate keywords instead of one parameterized keyword with `[Arguments]`.

**Experiment validation**: Experiment 5 confirmed that `[Arguments]` keywords work alongside embedded args in the same suite and with BDD prefixes.

**Fix**: Detect consecutive Fill steps targeting a common form context and generate a single keyword:

```python
# Detect: Fill Text #first-name + Fill Text #last-name + Fill Text #zip
# Generate:
# the user enters checkout details
#     [Arguments]    ${first_name}    ${last_name}    ${zip}
#     Fill Text    id=first-name    ${first_name}
#     Fill Text    id=last-name    ${last_name}
#     Fill Text    id=postal-code    ${zip}
```

**Estimated effort**: 12-16 hours for all Phase 3 items.

---

## Experiment Evidence Summary

| # | Experiment | Result | Key Finding |
|---|-----------|--------|-------------|
| 1 | RF BDD prefix stripping | PASS | Given/When/Then/And/But all stripped natively |
| 2 | Embedded arguments | PASS | BDD prefix stripped before embedded arg matching |
| 3 | Multi-level embedded args | PASS | Multi-word values, keyword reuse across tests work |
| 4 | Current heuristic quality | 2/5 accurate | Malformed names, wrong grouping, CSS artifacts |
| 5 | Ideal BDD validation | PASS | Target output structure (9 keywords, [Arguments], embedded args) is valid RF |

---

## Priority Matrix

| # | Improvement | Phase | Impact | Risk | Effort | Priority |
|---|------------|-------|--------|------|--------|----------|
| 1.1 | Word-level embedded name | 1 | High | Low | 2h | P0 |
| 1.2 | Instrumentation filtering | 1 | High | Low | 2h | P0 |
| 1.3 | Fix "the the" double article | 1 | Medium | Trivial | 0.5h | P0 |
| 1.4 | CSS locator sanitization | 1 | Medium | Low | 1h | P1 |
| 2.1 | Phase-aware grouping | 2 | High | Medium | 4h | P1 |
| 2.2 | Max group size constraint | 2 | Medium | Low | 1h | P1 |
| 2.3 | Multi-step name improvement | 2 | High | Low | 3h | P1 |
| 3.1 | Two-pass embedded arg arch | 3 | High | High | 4h | P2 |
| 3.2 | Locator-aware embedded args | 3 | Transformative | Medium | 4h | P2 |
| 3.3 | [Arguments] keyword generation | 3 | Medium | Medium | 4h | P2 |
| 4.1 | `bdd_group` MCP instruction examples | 4 | Medium | Low | 2h | P1 |
| 4.2 | Variables section generation | 4 | Medium | Low | 3h | P2 |
| 4.3 | BDD naming convention reference | 4 | Medium | Low | 1h | P1 |

**Total estimated effort**: 30-40 hours across 4 phases.

---

## Expected Outcome

After all 3 phases, the same 17-step demoshop flow should produce:

```robot
*** Test Cases ***
DemoShop Purchase
    Given the user opens the demoshop
    When the user adds "Sauce Labs Backpack" to the cart
    Then the cart badge should show "1" items
    When the user adds "Sauce Labs Bike Light" to the cart
    Then the cart badge should show "2" items
    When the user opens the cart
    Then the cart should contain "Sauce Labs Backpack"
    And the cart should contain "Sauce Labs Bike Light"
    When the user proceeds to checkout
    And the user enters checkout details    Test    User    12345
    And the user completes the order
    Then the order confirmation is displayed

*** Keywords ***
the user opens the demoshop
    New Browser    chromium    headless=False
    New Page    https://demoshop.makrocode.de/

the user adds "${product}" to the cart
    Click    role=button[name="Add ${product} to cart"]

the cart badge should show "${count}" items
    Get Text    css=.shopping_cart_badge    ==    ${count}

the user opens the cart
    Click    role=link[name="Cart"]

the cart should contain "${product}"
    Get Text    css=.cart_list    contains    ${product}

the user proceeds to checkout
    Click    text=Checkout

the user enters checkout details
    [Arguments]    ${first_name}    ${last_name}    ${zip}
    Fill Text    id=first-name    ${first_name}
    Fill Text    id=last-name    ${last_name}
    Fill Text    id=postal-code    ${zip}

the user completes the order
    Click    text=Continue
    Click    text=Finish

the order confirmation is displayed
    Get Text    css=.complete-header    ==    Thank you for your order!
```

This output:
- Has 12 BDD steps (vs current 9) — more granular, each step is meaningful
- Has 9 keywords (vs current 5) — each with a clear domain name
- Uses 2 embedded arg keywords (`"${product}"`, `"${count}"`) for parameterization
- Uses 1 `[Arguments]` keyword for the checkout form
- Contains zero CSS artifacts, zero generic fallbacks, zero malformed names
- Is valid RF syntax (confirmed by Experiment 5)

---

### Phase 4: Agent Guidance & Polish (Low Risk)

#### 4.1 Concrete `bdd_group` MCP Instruction Examples

**Problem**: The `execute_step(bdd_group=..., bdd_intent=...)` parameters exist but **zero LLMs used them** in E2E tests. The MCP instructions mention `bdd_group` but lack concrete examples.

**Fix**: Update `discovery_first.txt`, standard, and detailed templates with actionable examples:

```
BDD STEP GROUPING (for high-quality build_test_suite output):
When executing steps that belong to a single behavior, annotate them:

  execute_step(keyword="Click", arguments=["role=link[name='Products']"],
               bdd_group="navigate to products", bdd_intent="when")

  execute_step(keyword="Click", arguments=["button[aria-label='Add X to cart']"],
               bdd_group="add product to cart", bdd_intent="when")

  execute_step(keyword="Get Text", arguments=["[data-cart-count]", "==", "1"],
               bdd_group="verify cart count", bdd_intent="then")

Each unique bdd_group value becomes one keyword in the Keywords section.
Use domain language: bdd_group="add product to cart" NOT "click add button"
```

#### 4.2 Variables Section Generation

**Problem**: Hardcoded URLs, emails, names appear as literals in keyword bodies instead of being extracted to a `*** Variables ***` section.

**Fix**: After BDD transformation, scan keyword step arguments for:
- URLs (`https://...`) → `${BASE_URL}`
- Email addresses → `${TEST_EMAIL}`
- Repeated literal values across keywords → `${VARIABLE_NAME}`

```robot
*** Variables ***
${BASE_URL}       https://demoshop.makrocode.de/
${TEST_EMAIL}     test.user@example.com

*** Keywords ***
the user opens the demoshop
    New Browser    chromium    headless=False
    New Page    ${BASE_URL}
```

#### 4.3 BDD Naming Convention Reference in Instructions

**Fix**: Add a naming convention table to MCP instructions:

```
BDD Keyword Naming Conventions:
  Given: "the [noun] is [state]"         → "the demoshop is open"
  When:  "the user [verb]s [object]"     → "the user adds \"Backpack\" to the cart"
  Then:  "the [noun] should [condition]" → "the cart should contain 2 items"
  And:   Same pattern as preceding       → "And the total should be $98.50"
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Regression in simple (3-step) BDD scenarios | Keep existing simple-case path as fallback; comprehensive existing test suite (80+ tests) |
| Phase detection false positives | Phase indicators are conservative (exact word match); fallback to current grouping when no phase detected |
| Embedded arg detection too aggressive | Require 2+ keywords with identical structure before merging; never merge single-step keywords that differ in multiple arg positions |
| [Arguments] generation conflicts with embedded args | Clear rule: embedded args for keyword name parameterization, [Arguments] for multi-value keywords |

---

## Implementation Plan

1. **Phase 1** (P0, 1 sprint): Quick wins — word-level names, instrumentation filter, double-article fix, CSS sanitization
2. **Phase 2** (P1, 1 sprint): Semantic grouping — phase detection, max group size, improved naming
3. **Phase 3** (P2, 1-2 sprints): Intelligence — two-pass architecture, locator-aware embedding, [Arguments] generation

Each phase should include:
- Implementation in `test_builder.py`
- Unit tests with realistic e2e scenarios (not just 2-3 step happy paths)
- Re-run Experiment 4 as regression benchmark
- Update MCP instruction templates if naming conventions change

---

## References

- Experiment results: `tests/e2e/metrics/bdd_e2e_20260317/bdd_experiments.md`
- E2E evaluation: `tests/e2e/metrics/bdd_e2e_20260317/EVALUATION.md`
- Current BDD report: `docs/analysis/bdd-quality-improvement-report.md`
- Robot Framework BDD docs: https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#behavior-driven-style
