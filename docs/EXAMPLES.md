# rf-mcp Examples

Worked, copy-pasteable flows for driving Robot Framework through rf-mcp.

rf-mcp is not a Robot Framework library — it is an MCP server. Its "keywords"
are its MCP tools. Your agent discovers real RF keywords, executes them live
against Browser, SeleniumLibrary, RequestsLibrary, AppiumLibrary, DatabaseLibrary
or PlatynUI, watches what happens, and only then writes a `.robot` file. No
guessing, no dead suites. Test. Automate. Build.

Every example below shows three things: a short intro, **the flow** (which tools
to call, in order), and **the result** (the `.robot` snippet rf-mcp generates).
The flow is agent-facing — call the tools, don't hand-write JSON-RPC.

---

## The loop

One rhythm underlies all of them. Learn it once, reuse it everywhere:

| Step | Tool | Why |
|------|------|-----|
| 1. Set up | `manage_session(action="init", ...)` | Create a session, load libraries. Save the returned `session_id`. |
| 2. Get your bearings | `get_locator_guidance(library=...)` | Learn the selector/response syntax for the library *before* you touch a page. |
| 3. Discover | `find_keywords(query=...)` | Don't invent keyword names. Ask. |
| 4. Act | `execute_step` / `intent_action` / `execute_batch` | Run one keyword at a time, live. Capture results with `assign_to`. |
| 5. Look | `get_session_state(sections=["page_source"])` | Read the real DOM / ARIA tree to find real locators when something's off. |
| 6. Build | `build_test_suite(..., output_path=...)` | Turn the successful steps into a `.robot` file on disk. |

`analyze_scenario` is a fine step 1 too — hand it a plain-language description and
it creates the session and recommends libraries in one shot. Use whichever you
prefer; the rest of the loop is identical.

> **Saving suites.** Always pass `output_path` to `build_test_suite`. Never write
> the returned `rf_text` through the `Create File` keyword — Robot Framework
> resolves `${variables}` and expands `\n`/`\t` inside the argument and quietly
> corrupts the file. `output_path` does plain byte-for-byte file I/O.

---

## 1. Web with Browser (Playwright)

The saucedemo login, done with the Browser library. We log in with valid
credentials and assert we landed on the Products page.

**The flow**

1. `manage_session(action="init", session_id="web", libraries=["Browser", "BuiltIn"])`
2. `get_locator_guidance(library="browser")` — Browser uses Playwright selectors:
   `#id`, `css=…`, `text=Login`, `xpath=//…`. Plain selectors default to CSS.
3. `execute_step(keyword="New Browser", arguments=["chromium", "headless=False"], session_id="web")`
4. `execute_step(keyword="New Page", arguments=["https://www.saucedemo.com/"], session_id="web")`
5. `intent_action(intent="fill", target="#user-name", value="standard_user", session_id="web")`
6. `intent_action(intent="fill", target="#password", value="secret_sauce", session_id="web")`
7. `intent_action(intent="click", target="#login-button", session_id="web")`
8. `intent_action(intent="extract", target=".title", mode="text", assign_to="heading", session_id="web")`
9. `execute_step(keyword="Should Be Equal", arguments=["${heading}", "Products"], session_id="web")`
10. `build_test_suite(test_name="Saucedemo Login", session_id="web", output_path="/abs/path/saucedemo_login.robot")`

`intent_action` is library-agnostic — `intent="click"` resolves to Browser's
`Click`, Selenium's `Click Element`, or Appium's `Click Element` depending on the
session's active library. If you'd rather name keywords directly, `execute_step`
does the same job; `find_keywords(query="fill a text field", session_id="web")`
tells you what to call.

> `intent="navigate"` has a nicety: if no browser or page is open yet, it opens
> one and retries, so you can skip the explicit `New Browser`/`New Page` dance.

**The result**

```robotframework
*** Settings ***
Library    Browser

*** Test Cases ***
Saucedemo Login
    New Browser    chromium    headless=False
    New Page    https://www.saucedemo.com/
    Fill Text    \#user-name    standard_user
    Fill Text    \#password    secret_sauce
    Click    \#login-button
    ${heading}=    Get Text    .title
    Should Be Equal    ${heading}    Products
    [Teardown]    Close Browser
```

---

## 2. Web with SeleniumLibrary

Same site, same story, different engine. SeleniumLibrary has its own locator
grammar (`id:`, `name:`, `css:`, `xpath:`, `link:`), so ask for its guidance —
the syntax is not interchangeable with Browser's.

**The flow**

1. `manage_session(action="init", session_id="sel", libraries=["SeleniumLibrary", "BuiltIn"])`
2. `get_locator_guidance(library="selenium")` — note the `strategy:value` form (`id:user-name`).
3. `execute_step(keyword="Open Browser", arguments=["https://www.saucedemo.com/", "chrome"], session_id="sel")`
4. `execute_step(keyword="Input Text", arguments=["id:user-name", "standard_user"], session_id="sel")`
5. `execute_step(keyword="Input Text", arguments=["id:password", "secret_sauce"], session_id="sel")`
6. `execute_step(keyword="Click Button", arguments=["id:login-button"], session_id="sel")`
7. `execute_step(keyword="Get Text", arguments=["css:.title"], assign_to="heading", session_id="sel")`
8. `execute_step(keyword="Should Be Equal", arguments=["${heading}", "Products"], session_id="sel")`
9. `build_test_suite(test_name="Saucedemo Login Selenium", session_id="sel", output_path="/abs/path/saucedemo_selenium.robot")`

Selenium Manager fetches the browser driver for you — no extra install step. If a
locator misses, `get_session_state(session_id="sel", sections=["page_source"])`
hands you the live DOM to pick a real `id` or `css`.

**The result**

```robotframework
*** Settings ***
Library    SeleniumLibrary

*** Test Cases ***
Saucedemo Login Selenium
    Open Browser    https://www.saucedemo.com/    chrome
    Input Text    id:user-name    standard_user
    Input Text    id:password    secret_sauce
    Click Button    id:login-button
    ${heading}=    Get Text    css:.title
    Should Be Equal    ${heading}    Products
    [Teardown]    Close Browser
```

---

## 3. API with RequestsLibrary

No browser, no DOM — just requests and assertions. We use the public
[restful-booker](https://restful-booker.herokuapp.com) API: read the booking
list, create a booking with a JSON body, and pull the new `bookingid` out of the
response.

Call `get_locator_guidance(library="requests")` **first**. It is the API cookbook,
and it saves you the single biggest time sink in API testing — rediscovering
response access one failed `Evaluate` at a time. The rules it hands you:

- `Create Session` with an alias + base URL, then `GET/POST/… On Session` with a
  **relative** path.
- On-Session keywords **return** the response — capture it with `assign_to`.
  Read `${resp.status_code}`, `${resp.json()}`, a field via `${resp.json()["bookingid"]}`.
- Assert status with `Status Should Be    200    ${resp}`, not an `Evaluate`
  equality.
- Send a JSON body with `json=<dict>`, not `data=`. Build the dict inline with
  `${{ … }}` — a real Python literal, so `True`/`False`/`None` and nesting work.

**The flow**

1. `manage_session(action="init", session_id="api", libraries=["RequestsLibrary", "Collections", "BuiltIn"])`
2. `get_locator_guidance(library="requests")`
3. `execute_step(keyword="Create Session", arguments=["rb", "https://restful-booker.herokuapp.com"], session_id="api")`
4. `execute_step(keyword="GET On Session", arguments=["rb", "/booking"], assign_to="resp", session_id="api")`
5. `execute_step(keyword="Status Should Be", arguments=["200", "${resp}"], session_id="api")`
6. `execute_step(keyword="POST On Session", arguments=["rb", "/booking", 'json=${{ {"firstname": "Jane", "lastname": "Doe", "totalprice": 111, "depositpaid": True, "bookingdates": {"checkin": "2024-01-01", "checkout": "2024-01-05"}} }}'], assign_to="resp", session_id="api")`
7. `execute_step(keyword="Status Should Be", arguments=["200", "${resp}"], session_id="api")`
8. `execute_step(keyword="Set Variable", arguments=['${resp.json()["bookingid"]}'], assign_to="booking_id", session_id="api")`
9. `build_test_suite(test_name="Create And Read Booking", session_id="api", output_path="/abs/path/restful_booker.robot")`

> Need to assert a *non*-2xx response (e.g. a deleted booking returns 404)? Pass
> `expected_status=404` so the error response is returned instead of raising.
> PUT/DELETE on restful-booker need the auth token as a cookie header —
> `headers=${{"Cookie": "token=" + $token}}` after capturing `${resp.json()["token"]}`
> from `/auth`.

**The result**

```robotframework
*** Settings ***
Library    RequestsLibrary
Library    Collections

*** Test Cases ***
Create And Read Booking
    Create Session    rb    https://restful-booker.herokuapp.com
    ${resp}=    GET On Session    rb    /booking
    Status Should Be    200    ${resp}
    ${resp}=    POST On Session    rb    /booking
    ...    json=${{ {"firstname": "Jane", "lastname": "Doe", "totalprice": 111, "depositpaid": True, "bookingdates": {"checkin": "2024-01-01", "checkout": "2024-01-05"}} }}
    Status Should Be    200    ${resp}
    ${booking_id}=    Set Variable    ${resp.json()["bookingid"]}
```

---

## 4. Mobile with Appium

Same loop, an emulator instead of a browser. AppiumLibrary drives a running
Appium server (start it yourself; rf-mcp doesn't manage it). Ask
`get_locator_guidance(library="appium")` for the strategy zoo — `accessibility_id=`,
`id=`, `xpath=`, plus platform-specific `android=`/`ios=`.

**The flow**

1. `manage_session(action="init", session_id="mob", libraries=["AppiumLibrary", "BuiltIn"])`
2. `get_locator_guidance(library="appium")`
3. `execute_step(keyword="Open Application", arguments=["http://localhost:4723", "platformName=Android", "deviceName=emulator-5554", "app=/abs/path/SauceLabs.apk", "automationName=UiAutomator2"], session_id="mob")`
4. `intent_action(intent="fill", target="accessibility_id=test-Username", value="standard_user", session_id="mob")`
5. `intent_action(intent="click", target="accessibility_id=test-LOGIN", session_id="mob")`
6. `build_test_suite(test_name="Mobile Login", session_id="mob", output_path="/abs/path/mobile_login.robot")`

**The result**

```robotframework
*** Settings ***
Library    AppiumLibrary

*** Test Cases ***
Mobile Login
    Open Application    http://localhost:4723    platformName=Android
    ...    deviceName=emulator-5554    app=/abs/path/SauceLabs.apk    automationName=UiAutomator2
    Input Text    accessibility_id=test-Username    standard_user
    Click Element    accessibility_id=test-LOGIN
    [Teardown]    Close Browser
```

---

## 5. Desktop with PlatynUI

Native desktop GUIs on Linux and Windows, driven by `PlatynUI.BareMetal` (the Rust core).
The surest way in is `analyze_scenario(context="desktop")` — an explicit
`context="desktop"` *deterministically* forces a native PlatynUI session, so
scenario wording can't accidentally route you to Appium.

PlatynUI locators are descriptor XPaths against the accessibility tree, e.g.
`/app:*[@Name='myapp']//control:Button[@Name='OK']`. Get the syntax from
`get_locator_guidance(library="PlatynUI.BareMetal")`, and inspect the live tree
with `get_session_state(sections=["ui_tree"])`.

**The flow**

1. `analyze_scenario(scenario="Type a note into the Text Editor and save", context="desktop")` — save the `session_id`.
2. `get_locator_guidance(library="PlatynUI.BareMetal")`
3. `get_session_state(session_id="<id>", sections=["ui_tree"], elements_of_interest=["Text Editor"])` — find real `@Name` values.
4. `execute_step(keyword="Keyboard Type", arguments=["/app:*[@Name='Text Editor']//control:Text", "Hello from rf-mcp"], session_id="<id>")`
5. `execute_step(keyword="Pointer Click", arguments=["/app:*[@Name='Text Editor']//control:Button[@Name='Save']"], session_id="<id>")`
6. `build_test_suite(test_name="Edit And Save Note", session_id="<id>", output_path="/abs/path/desktop_note.robot")`

**The result**

```robotframework
*** Settings ***
Library    PlatynUI.BareMetal

*** Test Cases ***
Edit And Save Note
    Keyboard Type    /app:*[@Name='Text Editor']//control:Text    Hello from rf-mcp
    Pointer Click    /app:*[@Name='Text Editor']//control:Button[@Name='Save']
```

> Desktop is real input on a real desktop. On Linux, rf-mcp REFUSES desktop input by default unless the bound `DISPLAY` is provably isolated (marker `ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY` + `ROBOTMCP_PLATYNUI_ISOLATED_XPID`); to drive your active session set `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1` or `ROBOTMCP_PLATYNUI_SAFETY_GUARD=warn`. On Windows the guard allows the active desktop by default (opt in to strict isolation with `ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED=1`). The batch and recovery tooling is
> deliberately cautious here — a click or keystroke that provably never fired is
> retried; anything else is recorded rather than blindly re-fired.

---

## 6. BDD-style suite

Want a suite that reads like a specification? Group your steps into behavioral
keywords as you go, then generate with `bdd_style=True`. The Given/When/Then
prefixes live in the test case; the locators hide in a `*** Keywords ***` section.

**The rule:** never hang `Given`/`When`/`Then` on a raw library keyword like
`Click`. Attach them to *behavioral* keywords ("the demoshop is open"), and let
rf-mcp cluster the underlying steps beneath them.

**The flow** (against `https://demoshop.makrocode.de/`)

1. `manage_session(action="init", session_id="bdd", libraries=["Browser", "BuiltIn"])`
2. `execute_step(keyword="New Page", arguments=["https://demoshop.makrocode.de/"], session_id="bdd", bdd_group="the demoshop is open", bdd_intent="given")`
3. `execute_step(keyword="Click", arguments=["text=Add to cart"], session_id="bdd", bdd_group="the user adds a product to the cart", bdd_intent="when")`
4. `intent_action(intent="extract", target=".cart-count", mode="text", assign_to="count", session_id="bdd")`
5. `execute_step(keyword="Should Be Equal", arguments=["${count}", "1"], session_id="bdd", bdd_group="the cart holds one item", bdd_intent="then")`
6. `build_test_suite(test_name="Demoshop Purchase", session_id="bdd", bdd_style=True, output_path="/abs/path/demoshop_bdd.robot")`

Steps sharing a `bdd_group` collapse into one behavioral keyword; `bdd_intent`
assigns its Given/When/Then prefix.

**The result**

```robotframework
*** Test Cases ***
Demoshop Purchase
    Given the demoshop is open
    When the user adds a product to the cart
    Then the cart holds one item

*** Keywords ***
the demoshop is open
    New Page    https://demoshop.makrocode.de/

the user adds a product to the cart
    Click    text=Add to cart

the cart holds one item
    ${count}=    Get Text    .cart-count
    Should Be Equal    ${count}    1
```

---

## 7. Data-driven suite

One flow, many rows of data. rf-mcp builds a `Test Template` suite where each
named row becomes its own test case with its own pass/fail. Perfect for the
saucedemo login matrix — valid, locked-out, wrong password.

**The flow**

1. `manage_session(action="init", session_id="dd", libraries=["SeleniumLibrary", "BuiltIn"])`
2. Execute the flow once to prove the template keyword works (login + assert),
   grouping the steps so they become the template body.
3. `manage_session(action="start_test", session_id="dd", test_name="Valid User", template="Verify Login")`
4. `manage_session(action="add_data_row", session_id="dd", args=["standard_user", "secret_sauce", "Products"])`
5. `manage_session(action="end_test", session_id="dd")`
6. Repeat 3–5 for each row (`Locked Out User`, `Invalid Password`).
7. `build_test_suite(test_name="Login Matrix", session_id="dd", data_driven_mode="suite_template", output_path="/abs/path/login_matrix.robot")`

`data_driven_mode="suite_template"` puts a `Test Template` in Settings and turns
each named row into a separate, individually-reported test case. Leave it at
`"auto"` and rf-mcp decides: named rows → `suite_template`, otherwise a
per-test `[Template]`.

**The result**

```robotframework
*** Settings ***
Library         SeleniumLibrary
Test Template   Verify Login

*** Test Cases ***          USERNAME            PASSWORD        EXPECTED
Valid User                  standard_user       secret_sauce    Products
Locked Out User             locked_out_user     secret_sauce    locked out
Invalid Password            standard_user       wrong_pass      Username and password do not match

*** Keywords ***
# NOT generated by build_test_suite in suite_template mode — supply Verify Login yourself
Verify Login
    [Arguments]    ${username}    ${password}    ${expected}
    Open Browser    https://www.saucedemo.com/    chrome
    Input Text    id:user-name    ${username}
    Input Text    id:password    ${password}
    Click Button    id:login-button
    Page Should Contain    ${expected}
    [Teardown]    Close Browser
```

---

## Where to go next

- **Validate before you trust it:** `run_test_suite(session_id=..., mode="dry")`
  dry-runs the generated suite; `mode="full"` executes it.
- **Fewer round-trips:** `execute_batch` runs many keywords in one call, with
  `${STEP_N}` chaining between them and recovery on failure. `resume_batch`
  picks up from a failure point after you insert a fix.
- **When you're stuck:** `get_session_state(sections=["page_source"])` for the
  live DOM/ARIA tree, `find_keywords` when a keyword name won't come to you.

Built by the community, for the community. Now go automate something.
