# BDD Test Generation Quality Research Report

Date: 2026-03-17
Scope: BDD best practices, RF BDD patterns, robotmcp gap analysis, improvement proposals

---

## 1. BDD Best Practices Summary

### 1.1 Foundational Principles (Dan North, Cucumber, SpecFlow)

BDD originated from Dan North's insight that test-driven development improves when
tests are written in the language of the business domain rather than implementation
language. The core principle is that **scenarios should describe behavior visible to
stakeholders, not the mechanisms used to implement that behavior**.

Key tenets:

- **Declarative over imperative**: Steps should describe *what* happens, not *how*.
  "The user logs in" is declarative; "The user types admin into the username field
  and types password123 into the password field and clicks the login button" is
  imperative. BDD steps should lean declarative.

- **Ubiquitous language**: Steps use the shared vocabulary of the project domain.
  A checkout scenario uses terms like "cart", "order", "shipping address" -- not
  "click", "fill text", "CSS selector".

- **Single level of abstraction per scenario**: A scenario should not mix high-level
  business intent with low-level UI mechanics. Either the entire scenario speaks at
  the business level, or you are not doing BDD.

- **One behavior per step**: Each Given/When/Then step represents exactly one
  meaningful behavior. "The user adds a product and completes checkout" is two
  behaviors crammed into one step.

- **Given = precondition, When = action, Then = observable outcome**: This is not
  just syntax sugar. Given sets up state that already exists. When performs the
  action under test. Then asserts the externally visible outcome.

### 1.2 Step Naming Conventions

| Prefix | Purpose | Naming Pattern | Example |
|--------|---------|----------------|---------|
| Given  | Establish precondition | `the [noun] is [state]` | `the demoshop is open` |
| When   | Perform user action | `the user [verb]s [object]` | `the user adds "Backpack" to the cart` |
| Then   | Assert observable result | `the [noun] should [condition]` | `the cart should contain 2 items` |
| And    | Continue preceding type | Same pattern as preceding | `And the cart total should be $98.50` |
| But    | Negative continuation | `the [noun] should not [condition]` | `But the discount should not apply` |

### 1.3 Granularity Guidelines

**Good granularity**: Each keyword represents one user-perceivable action or one
verifiable assertion.

- "the user adds a product to the cart" -- one action
- "the cart should contain 2 items" -- one assertion
- "the user completes checkout" -- borderline; acceptable if checkout is a single
  click, but if it involves filling a form + clicking, it should be split into
  "the user fills in checkout information" and "the user places the order"

**Over-coarse**: Merging semantically distinct phases into one keyword. The worst
case from the E2E evaluation: 17 steps compressed into 3 keywords, where "the user
fills in the form" contained browse + add-to-cart + form-fill + submit.

**Over-fine**: Wrapping every single library call in its own keyword, producing
1-step keywords that add indirection without abstraction. "the element is hovered"
wrapping a single `Hover` call adds no clarity.

**Rule of thumb**: A BDD keyword should have 1-5 implementation steps. Single-step
keywords are fine when they genuinely represent a meaningful behavior (e.g.,
"the user opens the cart" wrapping `Click a[href="/cart"]`). Keywords with 8+
steps should be scrutinized for splitting opportunities.

### 1.4 Common Anti-Patterns

| Anti-Pattern | Description | Impact |
|-------------|-------------|--------|
| BDD-prefix-on-library-keyword | `Given Click button[...]` | Violates abstraction; BDD prefixes belong on behavioral keywords only |
| Locator-in-test-case | Test case body contains CSS/XPath selectors | Unreadable to non-developers; maintenance nightmare |
| Catch-all assertion keyword | "the expected results are verified" for all assertions | Loses domain meaning; what exactly is being verified? |
| Catch-all action keyword | "the user clicks the element" for all clicks | Loses domain meaning; which element and why? |
| Giant composite keyword | One keyword doing setup + actions + verification | Violates single-responsibility; untestable |
| Mixed abstraction levels | Some steps are BDD, others are raw library calls | Inconsistent; confusing to readers |
| No Keywords section | BDD prefixes used but no abstraction layer | Cosmetic BDD without substance |
| Naming by mechanism | "the user types into input#email" | Should be "the user enters their email" |
| Missing embedded arguments | Hardcoded values inside keywords instead of parameters | No reusability across similar scenarios |
| Assertion-as-navigation | `Get Text body` used as "then" but is actually just reading page text with no assertion | Not actually verifying anything |

### 1.5 The Declarative vs. Imperative Spectrum

BDD practitioners generally agree on a spectrum:

```
IMPERATIVE                                                DECLARATIVE
(how)                                                     (what)
|                                                         |
Fill Text #email "a@b.com"                                The user logs in
Fill Text #pass "secret"            The user enters        as "admin"
Click button[type=submit]           email "a@b.com"
Wait For Elements State ...         and password "secret"
                                    and clicks login
```

The sweet spot for automated tests is *slightly* declarative -- enough to hide
locators and implementation details, but specific enough to be unambiguous about
what behavior is being tested. "The user logs in" is too vague if we need to test
different credential combinations. "The user logs in as ${username}" is the sweet
spot.

---

## 2. Robot Framework BDD Capabilities

### 2.1 BDD Prefix Stripping (Core RF Behavior)

Robot Framework has native BDD support via prefix stripping in its keyword resolver.
The relevant code is in `robot.running.namespace.Namespace._get_bdd_style_runner()`:

1. RF checks if the keyword name matches the BDD prefix regexp (built from all
   loaded language prefixes).
2. If a prefix matches, RF strips it and searches for a keyword matching the
   remainder.
3. The runner is then wrapped to use the original (prefixed) name for display.

English prefixes: `Given`, `When`, `Then`, `And`, `But`.

This means:
```robot
Given the browser is open
```
will search for a keyword named `the browser is open` (without "Given").

**Implication for robotmcp**: The user MUST define a keyword `the browser is open`
in the Keywords section. RF does not auto-create it. The BDD prefix is purely
a display/organization mechanism.

### 2.2 Embedded Arguments (Core RF Feature)

RF supports embedded arguments in keyword names via the `EmbeddedArguments` class:

```robot
*** Keywords ***
the user adds "${product}" to the cart
    Click    button[aria-label="Add ${product} to cart"]
```

When called as:
```robot
When the user adds "Echo Speaker" to the cart
```

RF matches "Echo Speaker" against the `${product}` placeholder using a regex
(`.*?` by default), extracts the argument, and passes it to the keyword.

Key behaviors:
- Variables in the keyword name (`${...}`) become arguments
- Pattern matching is case-insensitive
- Multiple embedded args are supported: `the cart has ${count} ${unit}`
- Custom regex patterns allowed: `${count:\d+}` for digits only
- Type hints: `${count: int}` for automatic conversion (RF 7+)

**Implication for robotmcp**: Embedded arguments are the primary mechanism for
parameterized BDD keywords. The test_builder must generate keyword names with
`${arg}` placeholders when it detects repeated patterns differing only in
argument values.

### 2.3 RF BDD Best Practice Structure

The canonical RF BDD test structure:

```robot
*** Settings ***
Library    Browser

*** Variables ***
${BASE_URL}    https://demoshop.example.com/

*** Test Cases ***
Purchase Two Products And Complete Checkout
    Given the demoshop is open
    When the user adds "Echo Speaker" to the cart
    Then the cart should contain 1 item
    When the user adds "Focus Timer" to the cart
    Then the cart should contain 2 items
    When the user opens the cart
    Then the cart should show 2 products
    When the user completes checkout
    Then the order should be confirmed

*** Keywords ***
the demoshop is open
    New Browser    chromium    headless=${HEADLESS}
    New Page    ${BASE_URL}    wait_until=networkidle

the user adds "${product}" to the cart
    Click    button[aria-label="Add ${product} to cart"]

the cart should contain ${count} item
    Get Text    [data-cart-count]    ==    ${count}

the cart should contain ${count} items
    the cart should contain ${count} item

the user opens the cart
    Click    a[href="/cart"]

the cart should show ${count} products
    Get Element Count    main article    ==    ${count}

the user completes checkout
    Click    text="Proceed to checkout"
    Fill Text    #checkout-email    ${EMAIL}
    Fill Text    #checkout-name    ${FULL_NAME}
    Click    text="Place order"

the order should be confirmed
    Get Text    [role="alert"] >> nth=0    contains    confirmed!
```

Key patterns:
- Test case body contains ONLY Given/When/Then + behavioral keyword names
- ALL locators, CSS selectors, and technical details are in Keywords section
- Embedded arguments (`${product}`, `${count}`) enable parameterization
- Plural variants delegate to singular (`items` calls `item`)
- Multi-step behaviors grouped logically (checkout = fill + click)
- Setup keyword extracts URL from variable

### 2.4 Data-Driven BDD with [Template]

RF supports combining BDD with data-driven testing:

```robot
*** Test Cases ***
Add Product To Cart
    [Template]    Verify Product Can Be Added
    Echo Speaker     1
    Focus Timer      2
    Water Bottle     3

*** Keywords ***
Verify Product Can Be Added
    [Arguments]    ${product}    ${expected_count}
    When the user adds "${product}" to the cart
    Then the cart should contain ${expected_count} items
```

---

## 3. Gap Analysis: Current robotmcp vs. Best Practices

### 3.1 Heuristic Grouping Algorithm Deficiencies

The current `_group_by_heuristics()` method groups steps solely by intent
transitions (given -> when -> then). When intent stays the same across multiple
logically distinct actions, they all merge into one giant keyword.

**Problem demonstrated in E2E**: Claude's 17-step execution produced:
1. `New Browser` + `New Page` -> "given" group -> "the demoshop is open" (correct)
2. `Click` (products) + `Hover` + `Click` (add to cart) + `Click` (add second) + `Hover` + `Click` -> all "when" -> merged into "the user fills in the form" (WRONG)
3. `Get Text` + `Should Contain` -> "then" group -> "the expected results are verified" (WRONG)

The algorithm has no concept of **semantic boundaries** within a phase. Adding
product 1 to cart and adding product 2 to cart are separate behaviors, both "when",
but they should be separate keywords.

**Root cause**: `_classify_step_intent()` returns only 3 values (given/when/then).
The grouper flushes only on intent change. Consecutive "when" steps always merge.

### 3.2 Keyword Naming is Generic

`_generate_bdd_keyword_name()` produces these names for multi-step groups:

| Intent | Current output | Problem |
|--------|---------------|---------|
| when (multiple fill+click) | "the user fills in the form" | Does not describe which form or which action |
| when (multiple clicks) | "the user performs the action" | Completely generic |
| then (any assertion group) | "the expected results are verified" | Does not say what is being verified |
| when (1 click) | "the user clicks [locator]" | OK for single step, but locator text may be poor |
| then (1 get text) | "the [locator] shows the expected value" | Does not describe what value or what element |

The problem is worst for multi-step groups where the method falls through to
generic names like "the user fills in the form" or "the expected results are
verified" regardless of what the steps actually do.

### 3.3 Embedded Argument Detection is Late and Superficial

`_detect_embedded_args()` runs AFTER grouping and naming. It compares structurally
similar keywords and creates embedded arg placeholders. However:

1. The name template built by `_build_embedded_name()` uses string prefix/suffix
   matching on already-generic names. Result: `the user clicks ${value} t`
   (truncated, meaningless).

2. Detection happens at the keyword level, not the step level. If two "add to cart"
   actions are merged into one big "when" keyword, the individual click targets
   are buried and cannot be detected.

3. No awareness of the underlying step arguments. The embedded arg detection
   does not know that `button[aria-label="Add Echo Speaker to cart"]` and
   `button[aria-label="Add Focus Timer to cart"]` differ only in the product name.

### 3.4 Assertion Keywords are Bloated

Copilot's generated output shows a `the expected results are verified` keyword
with 14 lines of `Get Property` calls. This happens because the agent called
many `Get Property` steps for verification, and the heuristic grouper merged
them all into one assertion keyword.

The problem:
- Multiple distinct assertions (email field value, name field value, button
  disabled state, order ID) are conflated into one opaque "verified" keyword.
- The keyword name tells the reader nothing about what is being verified.
- If any single assertion fails, the keyword name gives no hint which one.

### 3.5 No Semantic Boundary Detection

The heuristic grouper has no concept of:

- **Page transitions**: Navigating from products page to cart page to checkout
  page. Each page transition should likely start a new keyword group.
- **Form submission**: Fill fields + click submit is a natural boundary.
- **Cart operations**: Add-to-cart + verify-cart-count is a natural pair.
- **Repeated patterns**: Adding product A and adding product B are the same
  pattern; they should produce a parameterized keyword, not be merged.

### 3.6 `bdd_group` Parameter Underutilized

The `execute_step(bdd_group="...", bdd_intent="...")` parameters exist but are
not used by any LLM in practice. The E2E tests show zero `bdd_group` annotations
in both Copilot and Claude runs. The MCP instructions mention `bdd_group` but do
not provide examples of how to use it or why it matters.

### 3.7 Discovery_first.txt BDD Guidance is Minimal

The discovery_first.txt template has 5 lines of BDD guidance (lines 37-46). It
says "ALWAYS create behavioral keywords" and "Good BDD: the user is on the login
page" but provides no structural examples, no naming conventions, and no
explanation of the Keywords section structure.

The standard template has more guidance (lines 496-503) but still lacks:
- Examples of proper BDD keyword structure
- Naming conventions for different intent types
- Explanation of when/how to use bdd_group
- Multi-scenario grouping guidance
- Embedded argument examples

### 3.8 No Page-Transition Awareness

The grouper does not detect URL changes or page navigations between steps. In
a multi-page flow (products -> cart -> checkout -> confirmation), steps on
different pages should generally not be merged into one keyword.

### 3.9 Summary of Gaps

| Gap | Severity | Root Cause |
|-----|----------|------------|
| Over-aggressive merging (17 -> 3) | Critical | Grouper only flushes on intent change |
| Generic keyword names | Critical | Naming falls through to catch-all strings |
| No semantic boundary detection | High | No awareness of pages, forms, repeated patterns |
| Bloated assertion keywords | High | All "then" steps merge regardless of target |
| Embedded args on garbage names | High | Detection runs after naming, on already-poor names |
| bdd_group unused by LLMs | Medium | Insufficient instruction + no examples |
| No locator-to-domain mapping | Medium | _humanize_locator extracts text but does not map to domain concepts |
| Single-page grouping assumption | Medium | No URL/page transition detection |

---

## 4. Specific Improvement Ideas

### 4.1 Improvement: Sub-Intent Classification

Add finer-grained intent classification that distinguishes within "when" and "then":

```
Current:  given | when | then
Proposed: given | navigate | interact | fill_form | assert_text | assert_count | assert_state
```

This enables the grouper to create boundaries between different sub-intents:

```python
# Proposed sub-intent taxonomy
NAVIGATE_KEYWORDS = {"click"}  # only when target is a link/navigation
FILL_KEYWORDS = {"fill text", "fill", "type text", "select from list by value"}
ASSERT_TEXT_KEYWORDS = {"get text"}  # with == or contains
ASSERT_COUNT_KEYWORDS = {"get element count"}
ASSERT_STATE_KEYWORDS = {"get property", "get attribute", "element should"}
INTERACT_KEYWORDS = {"click", "hover", "press keys"}
```

A `Click` on a navigation link (containing "href", "Products", "Cart", "Checkout")
triggers a sub-intent boundary. A `Click` on an action button stays in the current
group.

### 4.2 Improvement: Argument-Aware Pattern Detection

Before grouping, scan the step list for repeated structural patterns:

```python
def _detect_repeated_patterns(self, steps: List[TestCaseStep]) -> List[PatternGroup]:
    """Detect steps that differ only in argument values.

    Example: Click button[aria-label="Add X to cart"]
             Click button[aria-label="Add Y to cart"]
    These should become one embedded-arg keyword:
        the user adds "${product}" to the cart
    """
```

This runs BEFORE grouping, so the embedded arg detection influences keyword naming
directly rather than being a post-hoc fixup.

### 4.3 Improvement: Domain-Aware Keyword Naming

Replace the current generic naming with a locator-semantics-based approach:

```
Current:  "the user clicks the element"       (for Click role=link[name="Products"])
Proposed: "the user navigates to products"    (extracts "Products" and detects navigation)

Current:  "the expected results are verified"  (for Get Text [data-cart-count] == 2)
Proposed: "the cart should contain 2 items"   (extracts "cart" from locator, "2" from assertion)

Current:  "the user fills in the form"         (for Fill Text #email + Fill Text #name)
Proposed: "the user fills in checkout details"  (extracts "checkout" from page context or locator pattern)
```

Implementation: enhance `_generate_bdd_keyword_name` to:

1. For single-step clicks with `role=link[name="X"]` or `text="X"` or `href="X"`:
   -> "the user navigates to {X}"
2. For single-step clicks with `aria-label="Add X to cart"`:
   -> "the user adds {X} to the cart" (pattern: "Add ... to cart")
3. For Get Text with `==` operator:
   -> "the {locator_domain} should be {expected_value}"
4. For Get Element Count with `==`:
   -> "there should be {count} {locator_domain} elements"
5. For Fill Text groups:
   -> "the user fills in {common_field_context}" where context is extracted
      from the common parent (checkout-form, login-form, etc.)

### 4.4 Improvement: Max-Steps-Per-Keyword Limit

Add a configurable limit (default: 5) on the maximum number of steps per BDD
keyword. When a group exceeds this limit, the grouper should split it at
natural boundaries:

1. After any step whose argument contains a URL or navigation target
2. After a "submit" action (Click on button with "submit", "place order", "confirm")
3. Between repeated patterns (add product A, add product B)
4. Between form-fill and verification sequences

### 4.5 Improvement: Assertion Keyword Splitting

Instead of merging all "then" steps into one "the expected results are verified",
create separate assertion keywords by target:

```
Current:
    the expected results are verified
        Get Text    [data-cart-count]    ==    2
        Get Text    [role="alert"]    contains    confirmed!
        Get Property    button    disabled

Proposed:
    the cart should contain 2 items
        Get Text    [data-cart-count]    ==    2
    the order should be confirmed
        Get Text    [role="alert"]    contains    confirmed!
    the place order button should be disabled
        Get Property    button    disabled
```

Rule: Each assertion step with a distinct locator target gets its own keyword,
unless consecutive assertions share the same locator context (e.g., multiple
assertions on the same element).

### 4.6 Improvement: Richer bdd_group Instruction Guidance

Update MCP instructions with concrete bdd_group examples:

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
Use domain language, not technical language:
  GOOD: bdd_group="add product to cart"
  BAD:  bdd_group="click add button"
```

### 4.7 Improvement: Page-Transition Detection

Track page context changes during step execution. When the URL changes or a
navigation keyword is executed, mark a boundary:

```python
def _detect_page_boundaries(self, steps: List[TestCaseStep]) -> List[int]:
    """Return indices where page transitions occur."""
    boundaries = []
    for i, step in enumerate(steps):
        kw = (step.keyword or "").lower()
        args = step.arguments or []
        if kw in ("go to", "new page"):
            boundaries.append(i)
        elif kw == "click" and args:
            loc = str(args[0]).lower()
            if any(nav in loc for nav in ["href=", "role=link", "/cart", "/checkout"]):
                boundaries.append(i)
    return boundaries
```

Groups should not span page boundaries.

---

## 5. Embedded Argument Opportunities

### 5.1 Product Addition Pattern

**Current**: Two `Click` steps with different `aria-label` values are merged into
one opaque keyword.

**Opportunity**:
```robot
*** Keywords ***
the user adds "${product}" to the cart
    Click    button[aria-label="Add ${product} to cart"]
```

Detection: Look for Click steps where the locator contains a product name
that varies between instances while the locator structure remains the same.

### 5.2 Cart Count Verification Pattern

**Current**: `Get Text [data-cart-count] == 1` and `Get Text [data-cart-count] == 2`
are merged into one "the expected results are verified".

**Opportunity**:
```robot
*** Keywords ***
the cart should contain ${count} items
    Get Text    [data-cart-count]    ==    ${count}
```

Detection: Same keyword, same locator, different expected-value argument.

### 5.3 Form Field Fill Pattern

**Current**: Multiple `Fill Text` calls with different locators merged into
"the user fills in the form".

**Opportunity** (two levels of abstraction):

Level 1 - Per-field keywords:
```robot
the user enters "${value}" as their email
    Fill Text    #checkout-email    ${value}

the user enters "${value}" as their name
    Fill Text    #checkout-name    ${value}
```

Level 2 - Grouped form keyword with arguments:
```robot
the user fills in checkout with email "${email}" and name "${name}"
    Fill Text    #checkout-email    ${email}
    Fill Text    #checkout-name    ${name}
```

Level 2 is typically preferred as it keeps the test case concise while still
being parameterizable.

### 5.4 Navigation Pattern

**Current**: `Click role=link[name="Products"]` becomes "the user clicks products".

**Opportunity**:
```robot
the user navigates to "${page}"
    Click    role=link[name="${page}"]
```

Detection: Click steps where the target is a link with a `name` attribute.

### 5.5 Embedded Arg Detection Algorithm Improvements

The current `_detect_embedded_args()` matches by structural key
(keyword sequence + arg count). This misses cases where:

- The keyword is the same but the locator structure differs slightly
  (`button[aria-label="Add X"]` vs. `button[aria-label="Add Y"]`)
- The arg count is the same but different args vary
  (`Fill Text #email a@b` vs. `Fill Text #name John`)

Proposed improvement: Use locator-template matching. Extract the variable part
from locators:

```python
def _extract_locator_template(locator: str) -> Tuple[str, List[str]]:
    """Extract a template and variable parts from a locator.

    'button[aria-label="Add Echo Speaker to cart"]'
    -> ('button[aria-label="Add {} to cart"]', ['Echo Speaker'])
    """
```

Then group steps by template identity rather than just structural key.

---

## 6. Current vs. Desired Output Comparison

### 6.1 Copilot E2E Output (Current)

```robot
*** Test Cases ***
Demoshop Checkout Flow
    Given the demoshop is open
    Then the the element shows the expected value
    When the user clicks the element
    Then the the element shows the expected value
    When the user clicks the element
    ...

*** Keywords ***
the the element shows the expected value
    Get Text    body

the user clicks ${value} t
    Click    ${value}

the expected results are verified
    Get Text    body
    Get Url
    Get Title
    Get Property    xpath=(...)[1]    value
    Get Property    xpath=(...)[2]    value
    ...14 lines...
```

**Problems**: Double "the the", generic names, truncated embedded arg name,
`Get Text body` as assertion (reads entire page, asserts nothing), 14-line
catch-all assertion keyword.

### 6.2 Claude E2E Output (Current)

```robot
*** Test Cases ***
BDD Purchase Flow
    Given the demoshop is open
    When the user fills in the form
    Then the expected results are verified

*** Keywords ***
the demoshop is open
    New Browser    chromium    headless=False
    New Page    https://demoshop.makrocode.de/

the user fills in the form
    Click    role=link[name="Products"]
    Hover    role=button[name="Add Insight Smart Notebook to cart"]
    Click    role=button[name="Add Insight Smart Notebook to cart"]
    Hover    role=button[name="Add Focus Loop Timer to cart"]
    Click    role=button[name="Add Focus Loop Timer to cart"]

the expected results are verified
    Get Text    role=link[name="Cart Cart items"]
    Should Contain    ${cart_text}    2
```

**Problems**: Only 3 keywords for 17 steps. "The user fills in the form"
contains browsing + two add-to-cart actions (not a form fill at all). Most
checkout and verification steps are missing from the output. The entire flow
from cart -> checkout -> order confirmation was lost.

### 6.3 Desired Output

```robot
*** Settings ***
Documentation    DemoShop BDD checkout flow
Library          Browser
Test Tags        bdd    demoshop    checkout

*** Variables ***
${BASE_URL}         https://demoshop.makrocode.de/
${PRODUCT_1}        Insight Smart Notebook
${PRODUCT_2}        Focus Loop Timer
${EMAIL}            test.user@example.com
${FULL_NAME}        Test User
${ADDRESS}          12345 Test Street

*** Test Cases ***
Purchase Two Products And Complete Checkout
    Given the demoshop is open
    When the user navigates to products
    And the user adds "${PRODUCT_1}" to the cart
    Then the cart should contain 1 item
    When the user adds "${PRODUCT_2}" to the cart
    Then the cart should contain 2 items
    When the user opens the cart
    Then the cart should show 2 products
    When the user proceeds to checkout
    And the user fills in checkout details
    And the user places the order
    Then the order should be confirmed

*** Keywords ***
the demoshop is open
    New Browser    chromium    headless=False
    New Page    ${BASE_URL}    wait_until=networkidle

the user navigates to products
    Click    role=link[name="Products"]

the user adds "${product}" to the cart
    Click    role=button[name="Add ${product} to cart"]

the cart should contain ${count} item
    Get Text    [data-cart-count]    ==    ${count}

the cart should contain ${count} items
    the cart should contain ${count} item

the user opens the cart
    Click    role=link[name="Cart Cart items"]

the cart should show ${count} products
    Get Element Count    main article    ==    ${count}

the user proceeds to checkout
    Click    text="Proceed to checkout"

the user fills in checkout details
    Fill Text    #checkout-email    ${EMAIL}
    Fill Text    #checkout-name    ${FULL_NAME}
    Fill Text    #checkout-address    ${ADDRESS}

the user places the order
    Click    text="Place order"

the order should be confirmed
    Get Text    [role="alert"]    contains    confirmed!
```

### 6.4 Key Differences

| Aspect | Current (Claude) | Desired |
|--------|-----------------|---------|
| Keywords generated | 3 | 11 |
| Steps per keyword (avg) | 5.7 | 1.5 |
| Max steps in keyword | 5 | 3 |
| Embedded args | 0 | 2 (`${product}`, `${count}`) |
| Keyword name quality | Generic ("fills in the form") | Domain-specific ("adds product to cart") |
| Assertion specificity | 1 catch-all | 3 specific assertions |
| Variables section | Missing | 6 variables |
| Locators in test case | None (correct) | None (correct) |
| Domain language | Partial | Full |
| Page-transition awareness | None | Separate keywords per page |

---

## 7. Priority-Ranked Improvement Recommendations

### P0 (Critical) -- Fix Heuristic Grouper
- Add sub-intent classification to break "when" and "then" into finer categories
- Add max-steps-per-keyword limit (default 5)
- Detect page-transition boundaries (navigation clicks, Go To, New Page)
- Split assertion groups by locator target

### P1 (High) -- Improve Keyword Naming
- Extract domain names from locator attributes (aria-label, name, text)
- Use assertion values in "then" keyword names ("should contain 2 items")
- Detect navigation vs. action clicks for naming ("navigates to" vs. "clicks")
- Pattern-match common UI actions (add-to-cart, proceed-to-checkout)

### P2 (High) -- Move Embedded Arg Detection Before Grouping
- Scan for repeated locator patterns before grouping
- Create embedded-arg keywords directly from pattern detection
- Use the extracted arg name in the keyword name (not post-hoc prefix/suffix)

### P3 (Medium) -- Enhance MCP Instruction Guidance
- Add concrete bdd_group examples to standard and detailed templates
- Add naming convention reference (given/when/then patterns)
- Explain the relationship between execute_step annotations and build_test_suite output
- Show before/after examples of bad vs. good BDD output

### P4 (Medium) -- Variables Section Generation
- Extract hardcoded URLs, emails, names into Variables section
- Detect repeated literal values across steps
- Generate meaningful variable names from context

### P5 (Low) -- Plural Variant Generation
- When an embedded-arg keyword asserts a count, generate singular/plural variants
  ("should contain ${count} item" / "should contain ${count} items")

---

## 8. Appendix: Key Source File Locations

| File | Purpose |
|------|---------|
| `/home/many/workspace/rf-mcp/src/robotmcp/components/test_builder.py` | BDD transformation pipeline: `_transform_to_bdd_style`, `_group_by_heuristics`, `_classify_step_intent`, `_generate_bdd_keyword_name`, `_humanize_locator`, `_detect_embedded_args`, `_build_embedded_name`, `_generate_rf_text` |
| `/home/many/workspace/rf-mcp/src/robotmcp/domains/keyword_resolution/value_objects.py` | BddPrefix, EmbeddedPattern, EmbeddedMatch value objects |
| `/home/many/workspace/rf-mcp/src/robotmcp/domains/keyword_resolution/services.py` | BddPrefixService, EmbeddedMatcherService, DataSourceLoaderService |
| `/home/many/workspace/rf-mcp/src/robotmcp/domains/instruction/value_objects.py` | InstructionTemplate (standard, detailed, discovery_first) with BDD guidance text |
| `/home/many/workspace/rf-mcp/src/robotmcp/domains/instruction/templates/discovery_first.txt` | Discovery-first template with BDD section (lines 37-46) |
| `/home/many/workspace/rf-mcp/src/robotmcp/models/execution_models.py` | ExecutionStep.bdd_group/bdd_intent fields |
| `/home/many/workspace/rf-mcp/src/robotmcp/server.py` | execute_step bdd_group/bdd_intent params (~line 3399), BDD prefix stripping (~line 3648) |
| `/home/many/workspace/rf-mcp/.venv/lib/python3.13/site-packages/robot/running/arguments/embedded.py` | RF's EmbeddedArguments class |
| `/home/many/workspace/rf-mcp/.venv/lib/python3.13/site-packages/robot/running/namespace.py` | RF's `_get_bdd_style_runner` (prefix stripping) |
| `/home/many/workspace/rf-mcp/.venv/lib/python3.13/site-packages/robot/conf/languages.py` | RF's BDD prefix definitions per language |
| `/home/many/workspace/rf-mcp/tests/e2e/metrics/bdd_e2e_20260317/EVALUATION.md` | Latest E2E evaluation results |
| `/home/many/workspace/rf-mcp/.robotmcp_artifacts/art_964ed44d6687.txt` | Copilot-generated .robot file (demonstrates naming problems) |
| `/home/many/workspace/rf-mcp/docs/analysis/bdd-quality-improvement-report.md` | Earlier analysis with proposed solutions |
| `/home/many/workspace/rf-mcp/tests/unit/test_bdd_suite_generation.py` | Existing unit tests for BDD generation (Phase 3) |
| `/home/many/workspace/rf-mcp/tests/unit/test_bdd_step_grouping.py` | Existing unit tests for bdd_group/bdd_intent fields (Phase 2) |
| `/home/many/workspace/rf-mcp/tests/unit/test_bdd_data_driven.py` | Existing unit tests for data-driven BDD (Phase 4) |
