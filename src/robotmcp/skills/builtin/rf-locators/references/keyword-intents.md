# Keyword intents per library

Ported from rf-mcp 0.36.0. Legacy source: main:src/robotmcp/domains/intent/aggregates.py (`_builtin_*_mappings`, `_*_EXTRACT_KEYWORDS`, `_SELENIUM_SELECT_KEYWORDS`, `_builtin_navigate_fallbacks`); intent/adapters/mcp_tool.py (`_apply_nth_to_locator`); intent/adapters/locator_normalizer_adapter.py; intent/weak_model_guards.py; plugins/builtin/{browser,selenium}_plugin.py (`KEYWORD_ALTERNATIVES`); utils/hints.py.

The legacy `intent_action` tool mapped these intents: navigate, click, fill, hover, select, assert_visible, extract (mode text, attribute, count, value, url or title), wait_for, ensure_focused (PlatynUI only). No legacy mapping existed for upload, drag, scroll or screenshot.

## Browser
| Intent | Keyword line |
|---|---|
| navigate | `Go To    https://example.com` (use Go To, not New Page, inside an open browser) |
| click | `Click    <sel>`. To force: `Click With Options    <sel>    force=True` (Click has no force) |
| fill | `Fill Text    <sel>    <value>` |
| hover | `Hover    <sel>` |
| select | `Select Options By    <sel>    label    <value>` (or `value`, `index`, `text`) |
| assert_visible | `Get Element States    <sel>    contains    visible` (see the note below) |
| text / attribute | `Get Text    <sel>` and `Get Attribute    <sel>    href` |
| count / value | `Get Element Count    <sel>` and `Get Property    <sel>    value` |
| url / title | `Get Url` and `Get Title` |
| wait_for | `Wait For Elements State    <sel>    visible    timeout=10s` |
| nth | `<sel> >> nth=2` (zero-based) |

Note: legacy assert_visible emitted `Get Element States    <sel>    visible    ==    True`. That puts `visible` where Browser expects an assertion operator, so it is probably invalid. Use the form in the table instead.

## SeleniumLibrary
| Intent | Keyword line |
|---|---|
| navigate | `Go To    https://example.com` |
| click | `Click Element    <loc>` |
| fill | `Input Text    <loc>    <value>` |
| hover | `Mouse Over    <loc>` |
| select | `Select From List By Label    <loc>    <label>` (default). Also `... By Value` and `... By Index` |
| assert_visible | `Element Should Be Visible    <loc>` |
| text / attribute | `Get Text    <loc>` and `Get Element Attribute    <loc>    <attr>` |
| count / value | `Get Element Count    <loc>` and `Get Value    <loc>` |
| url / title | `Get Location` and `Get Title` |
| wait_for | `Wait Until Element Is Visible    <loc>    10s` |
| nth | CSS only: `css:li.item:nth-of-type(3)` (1-based). Legacy ignored nth for other locator types |

Select defaults to label. Legacy had an opt-in "auto" mode that chose value for numeric input, but numeric visible labels (years, amounts) make that guess unreliable.

## AppiumLibrary
| Intent | Keyword line |
|---|---|
| navigate | `Go To Url    https://example.com` |
| click / fill | `Click Element    <loc>` and `Input Text    <loc>    <value>` |
| hover, select | none (touch only, no dropdown keyword) |
| assert_visible | `Element Should Be Visible    <loc>` |
| text / attribute | `Get Text    <loc>` and `Get Element Attribute    <loc>    <attr>` |
| count | `Get Matching Xpath Count    <xpath>` |
| wait_for | `Wait Until Element Is Visible    <loc>    10s` |

## PlatynUI.BareMetal
| Intent | Keyword line |
|---|---|
| click | `Pointer Click    /app:*[@Name='myapp']//control:Button[@Name='OK']` |
| fill | `Keyboard Type    <desc>    <Ctrl+A>Hello<Enter>` (focuses the element, then types; chords work) |
| hover | `Pointer Move To    <desc>` |
| assert_visible, wait_for | `Query    <desc>    only_first=True` (retries for up to 30s, then raises ElementNotFoundError) |
| text / attribute | `Get Attribute    <desc>    Name` and `Get Attribute    <desc>    Bounds` |
| ensure_focused | `Activate Window    <Frame/Window/Dialog desc>` |
| navigate, select | not mapped |

## Wrong-library keyword names (from `KEYWORD_ALTERNATIVES`)
| In session | Wrong name | Use |
|---|---|---|
| Browser | Open Browser | `New Browser    chromium    headless=${True}` then `New Page    <url>` |
| Browser | Close All Browsers | `Close Browser    ALL` |
| Browser | Get Source | `Get Page Source` |
| Browser | Input Text | `Fill Text` |
| Browser | Click Element, Click Button | `Click` |
| Browser | Wait Until Element Is Visible | `Wait For Elements State    <sel>    visible` |
| Browser | Wait Until Page Contains Element | `Wait For Elements State    <sel>    attached` |
| Selenium | New Browser | `Open Browser    <url>    chrome` |
| Selenium | New Page | `Go To    <url>` |
| Selenium | New Context | configure via `Open Browser ... options=` |
| Selenium | Fill Text / Click | `Input Text` and `Click Element` |
| Selenium | Wait For Elements State | `Wait Until Element Is Visible` |
| Selenium | Get Page Source | `Get Source` |

## Other guards
- Weak models sometimes wrote absence as a literal `null`, `None`, `nil`, `undefined`, `nan`, `n/a` or an empty string. Never pass one of these as a locator or URL.
- Browser's `Evaluate JavaScript` takes a selector first: `Evaluate JavaScript    ${None}    () => document.title`. Passing only the JS body produces "Unsupported token '{' while parsing css selector".
- Navigate recovery:
  - Browser, "no browser open": `New Browser    headless=False`, then `New Page`.
  - Browser, "target closed": `New Page`.
  - Selenium, "No browser is open" or "invalid session id": `Open Browser    about:blank    chrome`.

RequestsLibrary steps: see the rf-requests skill (`skill://rf-requests/SKILL.md`).
