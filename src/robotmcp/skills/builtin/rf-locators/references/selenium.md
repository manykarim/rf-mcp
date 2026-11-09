# SeleniumLibrary locators

## Strategies

| Strategy | Example |
|---|---|
| id / name | `id:submit`, `name:username` (often the most reliable for form fields) |
| identifier (id or name) | `identifier:email` |
| css / xpath | `css:button.primary`, `xpath://button[text()='Save']` - always with the prefix |
| link text | `link:Click here`, `partial link:Click` |
| class / tag | `class:button-primary`, `tag:div` |
| data attribute | `data:id:my_id` |
| dom | `dom:document.images[5]` |
| WebElement | a variable returned by `Get WebElement` or `Execute Javascript` |

`=` and `:` both separate the strategy; `:` is the documented form.

`css:li:nth-of-type(3)` is 1-based and counts siblings of the same tag, not "the third match overall"; for repeated widgets scope by a stable container or use an XPath index such as `xpath:(//li)[3]`.

## Frames, windows, alerts

- `Select Frame    id:payframe` ... `Unselect Frame`. Put `Unselect Frame` in `[Teardown]`: while a frame is selected the main page is invisible. Nested frames: `Select Frame` once per level. Cross-origin frames work the same.
- Tabs and windows: `Switch Window    NEW`, `Switch Window    MAIN`.
- JavaScript alerts block every command: `Handle Alert    ACCEPT` or `Handle Alert    DISMISS`.
- Shadow DOM: CSS and XPath do not pierce shadow roots - see shadow-dom-iframes.md.

## Failure recipes

| Symptom | Fix |
|---|---|
| Click intercepted | `Scroll Element Into View    <loc>`; close the overlay; a JavaScript click (`Execute Javascript    arguments[0].click()    ARGUMENTS    ${el}`) only as a last resort |
| Not interactable | `Wait Until Element Is Visible` or `Wait Until Element Is Enabled`, then `Scroll Element Into View` |
| Stale element | `Wait Until Page Contains Element    <loc>    timeout=10s`, then locate it again |
| Global timeouts | `Set Selenium Timeout    30s` |

## Wrong-library names

| In a SeleniumLibrary suite | Use |
|---|---|
| `New Browser` / `New Page` | `Open Browser    <url>    chrome` / `Go To    <url>` |
| `Fill Text` / `Click` | `Input Text` / `Click Element` |
| `Wait For Elements State` | `Wait Until Element Is Visible` |
| `Get Page Source` | `Get Source` |

Never import Browser and SeleniumLibrary in the same suite.

Ported from the rf-mcp 0.36.0 locator guidance (`get_selenium_locator_guidance`, `KEYWORD_ALTERNATIVES`, hint rules).
