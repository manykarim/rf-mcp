---
name: rf-locators
description: Locator cookbook for Browser, SeleniumLibrary, AppiumLibrary and PlatynUI - inspect with rf_repl, pick locators that survive UI changes, shadow DOM and iframes, keyword per intent, library choice. Read before writing or repairing UI steps.
---

# Picking locators with rf-mcp

## Inspect first, never guess

In an `rf_repl` session with the page or app open:

| Library | Inspect | Then check a candidate |
|---|---|---|
| Browser | `Get Aria Snapshot    css=body` (or a narrower container such as `main`, `header`, `form`) | `Get Element Count    <locator>` must be `1` |
| SeleniumLibrary | `Get Source` (large) or `Get WebElements    css:...` | `Get Element Count    <locator>` |
| AppiumLibrary | `Get Source` | `Get Matching Xpath Count    <xpath>` |
| PlatynUI | `Query    /app:*` then a scoped `Query` | `Query    <descriptor>    only_first=True` |

The ARIA snapshot lists roles and accessible names (`button "Add to cart"`, `textbox "Email"`) - the most stable handles a page has. It shows open shadow content but each iframe only as `- iframe`: snapshot a frame through it with `Get Aria Snapshot    id=payframe >>> body`.

## Stability order (most stable first)

1. Dedicated test hooks: `[data-testid="checkout-submit"]`, `id=submit`
2. Role + accessible name: `role=button[name="Place order"]`, `role=textbox[name="Email"]`
3. Form semantics: `[name="email"]`, `label` text, `input[type="email"]`
4. Visible text: `text="Sign in"` (exact) / `text=Sign in` (substring, case-insensitive)
5. Structural CSS/XPath (`div > div:nth-child(3)`, generated class names such as `.css-1x2y3z`, `>> nth=2`) - last resort; they break when markup changes.

Scope inside a stable container instead of using `nth`: `form[action="/checkout"] >> role=textbox[name="Email"]`, `article:has-text("Headphones") >> role=button[name="Add to cart"]`.

Put every locator in one place (a `*** Variables ***` section or the resource keyword that owns the page) so a UI change is a one-line fix.

## Shadow DOM and iframes

| Situation | Browser | SeleniumLibrary |
|---|---|---|
| Open shadow root | CSS, `text=`, `role=` pierce it: `login-card >> role=button[name="Sign in"]`; XPath does not | CSS and XPath do not pierce: `${host}=    Get WebElement    css=login-card` then `Evaluate    $host.shadow_root.find_element("css selector", "#user")` |
| iframe | `id=payframe >>> id=card`; nested or cross-origin: `id=outer >>> id=inner >>> text=Confirm` | `Select Frame    id=payframe` ... `Unselect Frame` (in `[Teardown]`) |
| Closed shadow root (`attachShadow({mode: 'closed'})`) | invisible to selectors, `Get Aria Snapshot` and host `Get Text` | `shadow_root` fails |

Closed root, Browser - force roots open and recreate the element (its state resets; repeat after every navigation):

```
Evaluate JavaScript    ${None}
...    () => { const orig = Element.prototype.attachShadow; Element.prototype.attachShadow = function (o) { return orig.call(this, {...o, mode: 'open'}); }; for (const el of document.querySelectorAll('closed-counter')) { el.replaceWith(document.createElement(el.localName)); } }
Click       closed-counter >> role=button[name="Increment"]
Get Text    closed-counter >> output    ==    1
```

Closed root, SeleniumLibrary (Chrome) - force roots open before the page loads, then reload:

```
${lib}=    Get Library Instance    SeleniumLibrary
Evaluate    $lib.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "const o = Element.prototype.attachShadow; Element.prototype.attachShadow = function (i) { return o.call(this, {...i, mode: 'open'}); };"})
Reload Page
```

(Replace `closed-counter` with the host tag.) Nested roots, frames inside shadow roots, and reading a closed root without a reload: [shadow-dom-iframes.md](references/shadow-dom-iframes.md).

## Test data by identity, not position

Pick items by a stable name (`article:has-text("Aurora Neural Headphones")`), not by grid position (`article >> nth=2`): positions change when data or defects change, names do not.

## Screenshots

`Take Screenshot` (Browser) / `Capture Page Screenshot` (Selenium, Appium) save a file and return its path. Read the image only for what the tree cannot tell you (canvas, overlap, colour, charts) - see [visual-checks.md](references/visual-checks.md).

## References

- [browser.md](references/browser.md) - Browser selector syntax, strict mode, failure recipes
- [selenium.md](references/selenium.md) - SeleniumLibrary strategies, frames, windows, alerts
- [shadow-dom-iframes.md](references/shadow-dom-iframes.md) - open and closed shadow roots, nested and cross-origin iframes, for both libraries (every snippet is executed by tests/cookbook)
- [appium.md](references/appium.md) - AppiumLibrary strategies, Android vs iOS, contexts
- [platynui.md](references/platynui.md) - desktop locators over UIA / AT-SPI
- [keyword-intents.md](references/keyword-intents.md) - the keyword for click, fill, select, assert, wait... per library
- [library-chooser.md](references/library-chooser.md) - which library for which task; Browser vs SeleniumLibrary
- [visual-checks.md](references/visual-checks.md) - when a screenshot beats the DOM
