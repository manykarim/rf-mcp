# BDD Quality Improvement Report for RobotMCP

## 1. Problem Statement

When agents use robotmcp to generate BDD-style Robot Framework tests, the output violates fundamental BDD principles. The agent merely prepends Given/When/Then to **technical library keywords** instead of creating **behavioral abstractions**.

### Example of Current (Bad) Output

```robot
*** Test Cases ***
BDD Demoshop Checkout
    Given New Browser    chromium    headless=False
    And New Page    ${BASE_URL}    networkidle
    When Click    button[aria-label="Add Echo Conference Speaker to cart"]
    Then Get Text    [data-cart-count]    ==    1
    When Click    button[aria-label="Add Focus Loop Timer to cart"]
    Then Get Text    [data-cart-count]    ==    2
    When Click    a[href="/cart"]
    Then Get Element Count    main article    ==    2
    When Click    text="Proceed to checkout"
    And Fill Text    \#checkout-email    ${CHECKOUT_EMAIL}
    And Fill Text    \#checkout-name    ${CHECKOUT_NAME}
    When Click    text="Place order"
    Then Get Text    [role="alert"] >> nth=0    contains    confirmed!
```

### What's Wrong

| Issue | Explanation |
|-------|-------------|
| **BDD prefixes on library keywords** | `Given New Browser` is not a behavior — it's a technical setup step. BDD prefixes belong on **user-defined keywords** that describe behavior |
| **No `*** Keywords ***` section** | True BDD in RF requires a Keywords section where behavior is abstracted away from implementation |
| **Locators leak into test cases** | `button[aria-label="Add Echo Conference Speaker to cart"]` is an implementation detail. BDD test cases should read like plain English |
| **No reusability** | Every locator and technical detail is hardcoded in the test case. Cannot parameterize or reuse across tests |
| **Mixing abstraction levels** | BDD prefixes suggest high-level behavior, but the steps are low-level browser commands |
| **No data separation** | Product names, emails, addresses hardcoded as arguments, not variables or parameters |

---

## 2. What Good BDD in Robot Framework Looks Like

### 2.1 The RF BDD Pattern (Official)

Robot Framework's BDD support works by **prefix stripping**: when a keyword is called with `Given`, `When`, `Then`, `And`, or `But`, the prefix is stripped and the remaining text is matched against user keywords. This means **the user must define behavioral keywords**.

Reference: [RF User Guide — Behavior Driven Development](https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#behavior-driven-development)

### 2.2 Correct BDD Structure

```robot
*** Settings ***
Documentation    DemoShop checkout flow using BDD style
Library          Browser
Test Tags        bdd    demoshop    checkout

*** Variables ***
${BASE_URL}         https://demoshop.makrocode.de/
${PRODUCT_1}        Echo Conference Speaker
${PRODUCT_2}        Focus Loop Timer
${EMAIL}            jamie.product@example.com
${FULL_NAME}        Jamie Product
${ADDRESS}          123 Flow Street\nSan Francisco, CA

*** Test Cases ***
Purchase Two Products And Complete Checkout
    [Documentation]    Verify that a user can add products to cart and complete checkout
    Given the demoshop is open
    When the user adds "${PRODUCT_1}" to the cart
    Then the cart should contain 1 item
    When the user adds "${PRODUCT_2}" to the cart
    Then the cart should contain 2 items
    When the user opens the cart
    Then the cart should show 2 products
    When the user completes checkout
    Then the order should be confirmed

*** Keywords ***
the demoshop is open
    New Browser    chromium    headless=False
    New Page    ${BASE_URL}    wait_until=networkidle

the user adds "${product}" to the cart
    Click    button[aria-label="Add ${product} to cart"]

the cart should contain ${count} item
    [Documentation]    Verify cart badge count (works for "1 item" and "2 items")
    Get Text    [data-cart-count]    ==    ${count}

the cart should contain ${count} items
    [Documentation]    Plural variant
    Get Text    [data-cart-count]    ==    ${count}

the user opens the cart
    Click    a[href="/cart"]

the cart should show ${count} products
    Get Element Count    main article    ==    ${count}

the user completes checkout
    Fill Text    \#checkout-email    ${EMAIL}
    Fill Text    \#checkout-name    ${FULL_NAME}
    Fill Text    \#checkout-address    ${ADDRESS}
    Click    text="Place order"

the order should be confirmed
    Get Text    [role="alert"] >> nth=0    contains    confirmed!
```

### 2.3 Key Differences

| Aspect | Bad BDD | Good BDD |
|--------|---------|----------|
| Test case readability | `When Click button[aria-label="..."]` | `When the user adds "Echo Speaker" to the cart` |
| Locator placement | In test case body | In `*** Keywords ***` section |
| Abstraction | None — raw library calls | Behavioral keywords hide implementation |
| Reusability | None | Keywords reusable across tests |
| Embedded arguments | Not used | `"${product}"`, `${count}` enable parameterization |
| Data-driven ready | No | Variables + embedded args enable templating |
| Maintainability | Change locator → change every test | Change locator → change one keyword |

---

## 3. Root Cause Analysis in RobotMCP

### 3.1 The `build_test_suite` Tool Only Replays Steps

`build_test_suite` captures the sequence of `execute_step` calls and renders them as a flat list of test case steps. There is **no mechanism to**:

1. **Group steps into behavioral keywords** — The tool has no concept of "these 3 steps represent adding a product to cart"
2. **Generate a `*** Keywords ***` section** — `_generate_rf_text()` only renders `*** Settings ***`, `*** Variables ***`, and `*** Test Cases ***`
3. **Apply BDD prefixes to user keywords** — Even though `execute_step` strips BDD prefixes (ADR-019), `build_test_suite` doesn't add them back
4. **Create embedded argument keywords** — The tool doesn't detect repeated patterns that could be parameterized

### 3.2 The LLM Has No BDD Guidance

The MCP server instructions tell agents that "BDD prefixes are auto-stripped from keywords" but provide **no guidance on how to write proper BDD tests**. Specifically:

- No guidance to create a `*** Keywords ***` section
- No guidance to abstract behavior into user keywords
- No guidance on BDD naming conventions (`the user ...`, `the cart should ...`)
- No examples of proper BDD structure
- The `execute_step` flow naturally produces flat step sequences, not layered BDD

### 3.3 The Execution Model is Step-by-Step

RobotMCP's core loop is `execute_step` → `build_test_suite`. This inherently produces flat test cases because each `execute_step` call is a single library keyword. There is no way to:

- Execute a **group of steps** and name that group
- Mark step boundaries that correspond to BDD keywords
- Record the intent behind a group of steps

---

## 4. Proposed Solutions

### 4.1 Solution A: BDD-Aware `build_test_suite` (Server-Side, Recommended)

Add a `bdd_style` parameter to `build_test_suite` that transforms the flat step list into a proper BDD structure with a `*** Keywords ***` section.

**New parameter**: `bdd_style: bool = False`

**When `bdd_style=True`**:

1. **Group steps into behavioral clusters** using heuristics:
   - Navigation steps → "the [page] is open"
   - Click + assertion on same element → "the user [action]s [target]"
   - Fill + Click sequence → "the user completes [form]"
   - Assertion steps → "the [thing] should [condition]"

2. **Generate `*** Keywords ***` section** with:
   - Behavioral keyword names (lowercase, natural language)
   - Implementation steps (the actual library keywords with locators)
   - Embedded arguments for repeated patterns

3. **Rewrite test case** to use BDD prefixes (`Given`/`When`/`Then`/`And`) with behavioral keyword names

4. **Move locators out of test case** into keyword implementations

**Implementation sketch** for `test_builder.py`:

```python
class BddKeywordGroup:
    """A group of steps that form a behavioral keyword."""
    name: str                    # e.g., "the user adds "${product}" to the cart"
    steps: List[TestCaseStep]    # Implementation steps
    bdd_prefix: str              # "Given", "When", "Then", "And"
    embedded_args: Dict[str, str]  # e.g., {"product": "Echo Speaker"}

def _group_steps_into_bdd_keywords(steps: List[TestCaseStep]) -> List[BddKeywordGroup]:
    """Group flat steps into behavioral keyword clusters."""
    # Strategy 1: Use step annotations from execute_step (see 4.2)
    # Strategy 2: Heuristic grouping by keyword type
    # Strategy 3: LLM-assisted grouping via analyze_scenario
```

### 4.2 Solution B: Step Annotation in `execute_step` (Server-Side)

Add optional `bdd_group` and `bdd_intent` parameters to `execute_step` so the agent can annotate steps with behavioral intent during execution.

```python
async def execute_step(
    keyword: str,
    arguments: list = None,
    session_id: str = "",
    # New BDD annotations
    bdd_group: str = "",      # Group name, e.g., "add product to cart"
    bdd_intent: str = "",     # "given", "when", "then", "and", "but"
):
```

Steps with the same `bdd_group` value are collected into a single behavioral keyword at `build_test_suite` time.

**Pros**: Agent has full control over BDD structure
**Cons**: Requires agent to plan BDD groups upfront; adds parameters to every call

### 4.3 Solution C: BDD Instruction Enhancement (Prompt-Side, Quick Win)

Update the MCP server instructions to teach agents how to write proper BDD. This is the lowest-effort change with immediate impact.

**Add to MCP instructions (standard and detailed templates)**:

```
BDD STYLE GUIDELINES (when user requests BDD):
- NEVER put Given/When/Then on library keywords like Click, Fill Text, New Browser
- ALWAYS create a *** Keywords *** section with behavioral keyword names
- Behavioral keywords should read like English: "the user adds a product"
- Use embedded arguments for parameterization: the user adds "${product}"
- Put all locators and technical details INSIDE keyword implementations
- Test cases should contain ONLY Given/When/Then + behavioral keyword names
- Use Given for preconditions, When for actions, Then for assertions

BAD:  Given New Browser    chromium    headless=False
GOOD: Given the browser is open

BAD:  When Click    button[aria-label="Add item"]
GOOD: When the user adds "Echo Speaker" to the cart

The behavioral keyword "the user adds "${product}" to the cart" is defined
in *** Keywords *** with the actual Click step inside it.
```

### 4.4 Solution D: `build_bdd_suite` New Tool (Server-Side, Most Complete)

Create a dedicated `build_bdd_suite` tool that takes the step history AND a behavioral structure specification from the agent.

```python
async def build_bdd_suite(
    test_name: str,
    session_id: str = "",
    keywords: list = None,  # List of {name, steps, embedded_args}
    # e.g., [{"name": "the user adds \"${product}\" to the cart",
    #         "steps": [3, 4],  # step indices
    #         "embedded_args": {"product": "str"}}]
):
```

The agent provides the behavioral keyword definitions referencing step indices, and the tool generates the full BDD suite with `*** Keywords ***` section.

### 4.5 Solution E: Hybrid — Heuristic + Agent Post-Processing

1. `build_test_suite(bdd_style=True)` uses heuristics for initial grouping
2. Returns the grouped structure as a suggestion
3. Agent can refine the grouping by calling a `refine_bdd_keywords` tool
4. Final suite is generated from the refined structure

---

## 5. Recommended Implementation Plan

### Phase 1: Quick Win — MCP Instruction Enhancement (Solution C)

**Effort**: Small (update 3 template strings)
**Impact**: Immediate improvement in agent-generated BDD quality

1. Update `standard` template in `value_objects.py`
2. Update `detailed` template in `value_objects.py`
3. Update `discovery_first.txt` template
4. Add BDD examples to the MCP server instructions text

### Phase 2: Step Grouping in `execute_step` (Solution B, partial)

**Effort**: Medium (add parameter, store annotations)
**Impact**: Enables agent to communicate BDD intent

1. Add `bdd_group: str = ""` parameter to `execute_step`
2. Store group annotation in step history on `ExecutionSession`
3. Use group annotations in `build_test_suite` when generating RF text

### Phase 3: BDD-Aware Suite Generation (Solution A)

**Effort**: Large (new logic in test_builder.py)
**Impact**: Automatic BDD structure generation

1. Add `bdd_style: bool = False` parameter to `build_test_suite`
2. Implement `_group_steps_into_bdd_keywords()` with heuristic clustering
3. Implement `*** Keywords ***` section rendering in `_generate_rf_text()`
4. Generate embedded argument keywords for repeated patterns
5. Apply BDD prefix assignment (Given for setup, When for actions, Then for assertions)

### Phase 4: Data-Driven BDD Combination

**Effort**: Medium (extend existing template support)
**Impact**: Full BDD + data-driven support

1. Combine BDD keywords with `[Template]` support
2. Generate data-driven BDD tests: template keyword with Given/When/Then inside
3. Support companion data files (CSV/JSON) for parameterized BDD scenarios

---

## 6. BDD Best Practices Reference

### 6.1 Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| **Given** (precondition) | `the [noun] is [state]` | `the demoshop is open` |
| **When** (action) | `the user [verb]s [object]` | `the user adds "Backpack" to the cart` |
| **Then** (assertion) | `the [noun] should [condition]` | `the cart should contain 2 items` |
| **And/But** (continuation) | Same as the preceding type | `And the user fills in the email` |

### 6.2 Keyword Granularity Rules

- **One behavior per keyword**: "the user completes checkout" is too coarse if it includes 5 form fields + a click. Better: separate "fills checkout form" and "submits the order"
- **No technical details in test cases**: Locators, CSS selectors, XPaths should never appear in test case body
- **Embedded arguments for parameterization**: `the user adds "${product}" to the cart` not `the user adds item` with hardcoded product name inside
- **Keywords section mirrors test case**: Every Given/When/Then step in the test case has a corresponding keyword definition

### 6.3 RF-Specific BDD Patterns

```robot
# PATTERN 1: Embedded arguments for parameterized behavior
*** Keywords ***
the user adds "${product}" to the cart
    Click    button[aria-label="Add ${product} to cart"]

the cart should contain ${count} item
    Get Text    [data-cart-count]    ==    ${count}

the cart should contain ${count} items
    the cart should contain ${count} item


# PATTERN 2: Setup/teardown as Given
*** Test Cases ***
Checkout Flow
    [Setup]    Given the demoshop is open
    When the user adds "Backpack" to the cart
    Then the cart should contain 1 item
    [Teardown]    Close Browser


# PATTERN 3: Data-driven BDD with [Template]
*** Test Cases ***
Add Product To Cart
    [Template]    Verify Product Can Be Added
    Echo Conference Speaker    1
    Focus Loop Timer           2
    Cascade Water Bottle       3

*** Keywords ***
Verify Product Can Be Added
    [Arguments]    ${product}    ${expected_count}
    When the user adds "${product}" to the cart
    Then the cart should contain ${expected_count} items


# PATTERN 4: Resource file for cross-suite reuse
# demoshop_keywords.resource
*** Keywords ***
the demoshop is open
    New Browser    chromium    headless=${HEADLESS}
    New Page    ${BASE_URL}    wait_until=networkidle
```

### 6.4 Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Fix |
|-------------|-------------|-----|
| `Given Click    button` | BDD prefix on library keyword | Create user keyword |
| `Then Should Be Equal    ${x}    5` | Technical assertion as behavior | `Then the count should be 5` |
| Locators in test case body | Implementation detail leaks | Move to Keywords section |
| One giant keyword | "the user does everything" | Split into Given/When/Then steps |
| Mixing BDD and non-BDD | `Given setup` then `Click button` | Be consistent — all BDD or none |
| No `*** Keywords ***` section | BDD without abstraction layer | Always define behavioral keywords |

---

## 7. Impact Assessment

### Current State (ADR-019)

- BDD prefix stripping works correctly in `execute_step`
- Embedded argument matching works in `KeywordDiscovery`
- `build_test_suite` generates flat step sequences without Keywords section
- MCP instructions mention BDD support but don't guide proper usage

### After Phase 1 (Instruction Enhancement)

- Agents will understand BDD naming conventions
- Generated test cases will have `*** Keywords ***` sections (agent-authored)
- Locators will be inside keywords, not in test cases
- **Expected improvement**: 60-70% of BDD quality issues fixed by better prompting

### After Phase 3 (BDD-Aware Suite Generation)

- `build_test_suite(bdd_style=True)` auto-generates Keywords section
- Heuristic step grouping produces behavioral abstractions
- Embedded arguments auto-detected for repeated patterns
- **Expected improvement**: 90%+ of BDD quality issues eliminated

---

## 8. Appendix: Test Case Comparison

### Input Scenario
> Add two products to cart, verify counts, complete checkout

### Current Output (Bad BDD)
```
Given New Browser    chromium    headless=False
When Click    button[aria-label="Add Echo Conference Speaker to cart"]
Then Get Text    [data-cart-count]    ==    1
```
**Readability**: Low — requires understanding CSS selectors
**Maintainability**: Low — locator changes break test case

### Desired Output (Good BDD)
```
Given the demoshop is open
When the user adds "Echo Conference Speaker" to the cart
Then the cart should contain 1 item
```
**Readability**: High — reads like English specification
**Maintainability**: High — locator changes only affect Keywords section
