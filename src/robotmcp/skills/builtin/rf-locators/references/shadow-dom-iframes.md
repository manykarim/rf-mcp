# Shadow DOM and iframes

Every snippet here is executed against a fixture page with open, nested and closed shadow roots and same-origin, nested, cross-origin and shadow-hosted iframes (`tests/cookbook/browser_*.robot`, `tests/cookbook/selenium_*.robot`).

## Map the page first (Browser)

`Get Aria Snapshot    css=body` shows open shadow content, but each iframe only as `- iframe`, and nothing of closed shadow roots. Snapshot a frame through it: `Get Aria Snapshot    id=payframe >>> body`.

List shadow hosts and frames with their paths in one call:

```
${map}=    Evaluate JavaScript    ${None}
...    () => { const out = []; const walk = (root, path) => { for (const el of root.querySelectorAll('*')) { if (el.shadowRoot) { out.push(path + el.localName + ' [open shadow]'); walk(el.shadowRoot, path + el.localName + ' >> '); } else if (el.localName.includes('-') && !el.shadowRoot) { out.push(path + el.localName + ' [custom element: closed shadow or none]'); } if (el.localName === 'iframe') { out.push(path + 'iframe#' + el.id + ' src=' + el.getAttribute('src') + ' -> use >>>'); } } }; walk(document, ''); return out; }
```

## Browser library

| Situation | What works |
|---|---|
| Open shadow root, any depth | CSS, `text=` and `role=` pierce automatically: `css=#user`, `login-card >> role=button[name="Sign in"]` |
| XPath | does NOT pierce shadow roots - use CSS, role or text |
| iframe | `id=payframe >>> id=card` - without `>>>` the element is not found |
| Nested and cross-origin iframes | chain frames: `id=outer >>> id=inner >>> text=Confirm`; cross-origin works the same |
| iframe inside a shadow root | `css=shadow-frame >> iframe >>> id=coupon` |
| Shadow root inside an iframe | `id=cardframe >>> id=cardno` (pierces after the frame) |
| JavaScript in a frame or on a shadow host | `Evaluate JavaScript    id=payframe >>> id=card    (el) => el.value`; `Evaluate JavaScript    login-card    (host) => host.shadowRoot.querySelector('h2').textContent` |

### Closed shadow roots (`attachShadow({mode: 'closed'})`)

Selectors (`css=`, `text=`, `role=`) never enter them, they are missing from `Get Aria Snapshot`, and `Get Text` on the host does not include their text. Ask the developers for a test hook if you can. Otherwise force shadow roots open and recreate the element - its state resets, and you must run it again after every navigation or reload:

```
Evaluate JavaScript    ${None}
...    () => { const orig = Element.prototype.attachShadow; Element.prototype.attachShadow = function (o) { return orig.call(this, {...o, mode: 'open'}); }; for (const el of document.querySelectorAll('closed-counter')) { el.replaceWith(document.createElement(el.localName)); } }
Click       closed-counter >> role=button[name="Increment"]
Get Text    closed-counter >> output    ==    1
```

(Replace `closed-counter` with the host tag.)

## SeleniumLibrary

CSS and XPath locators do NOT pierce shadow roots. Get the host, then pass the inner WebElement as the locator:

```
${host}=    Get WebElement    css=login-card
${user}=    Evaluate    $host.shadow_root.find_element("css selector", "#user")
Input Text    ${user}    carol
Click Element    ${{ $host.shadow_root.find_element("css selector", "button") }}
```

Nested shadow roots - a deep query helper:

```
*** Variables ***
${DEEP_QUERY}    SEPARATOR=\n
...    const find = (root, sel) => {
...      const hit = root.querySelector(sel); if (hit) return hit;
...      for (const el of root.querySelectorAll('*')) { if (el.shadowRoot) { const f = find(el.shadowRoot, sel); if (f) return f; } }
...      return null; };
...    return find(document, arguments[0]);

*** Test Cases ***
Deep Button
    ${btn}=    Execute Javascript    ${DEEP_QUERY}    ARGUMENTS    button.deep
    Click Element    ${btn}
```

Frames:

- `Select Frame    id=payframe` ... `Unselect Frame`; put `Unselect Frame` in `[Teardown]` - while a frame is selected the main page is invisible.
- Nested frames: `Select Frame` once per level. Cross-origin frames work.
- iframe inside a shadow root: `${frame}=    Execute Javascript    ${DEEP_QUERY}    ARGUMENTS    iframe#couponframe`, then `Select Frame    ${frame}`.
- Shadow root inside an iframe: `Select Frame    id=cardframe` first, then `Get WebElement    css=card-input` and `shadow_root`.

### Closed shadow roots

`shadow_root` raises / returns nothing for a closed root. Chrome only, two options:

Force shadow roots open before the page loads, then reload (state resets):

```
${lib}=    Get Library Instance    SeleniumLibrary
Evaluate    $lib.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "const o = Element.prototype.attachShadow; Element.prototype.attachShadow = function (i) { return o.call(this, {...i, mode: 'open'}); };"})
Reload Page
```

Read without reloading (state kept): DevTools search pierces closed roots; interact by clicking at an offset inside the host.

```
*** Test Cases ***
Read Closed Shadow Via CDP Keeps State
    ${w}    ${h}=    Get Element Size    css=closed-counter
    Click Element At Coordinates    css=closed-counter    ${{ int(10 - $w / 2) }}    0
    Click Element At Coordinates    css=closed-counter    ${{ int(10 - $w / 2) }}    0
    ${html}=    Closed Shadow Outer HTML    output[aria-label="Count value"]
    Should Contain    ${html}    >2</output>

*** Keywords ***
Closed Shadow Outer HTML
    [Arguments]    ${css}
    ${lib}=    Get Library Instance    SeleniumLibrary
    ${drv}=    Set Variable    ${lib.driver}
    Evaluate    $drv.execute_cdp_cmd("DOM.getDocument", {"depth": -1, "pierce": True})
    ${search}=    Evaluate    $drv.execute_cdp_cmd("DOM.performSearch", {"query": $css})
    ${hits}=    Evaluate    $drv.execute_cdp_cmd("DOM.getSearchResults", {"searchId": $search["searchId"], "fromIndex": 0, "toIndex": 1})
    ${out}=    Evaluate    $drv.execute_cdp_cmd("DOM.getOuterHTML", {"nodeId": $hits["nodeIds"][0]})
    RETURN    ${out}[outerHTML]
```
