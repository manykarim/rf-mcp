# Browser library locators

## Selector syntax

| Form | Meaning |
|---|---|
| `id=submit`, `css=button.primary`, `text=Login`, `xpath=//input[@type='submit']` | explicit engines |
| `.btn > #login` | no prefix: CSS |
| `//...` or `..` | auto-detected XPath |
| `"Login"` | quoted string: exact text |
| `text=Login` / `text=/^Log(in|out)$/i` | case-insensitive substring / regex |
| `role=button[name="Sign in"]` | role + accessible name (pierces open shadow roots) |
| `data-testid=checkout` | test-id attribute engine |
| `button:text-is('Save')`, `:text('Login')` | Playwright text pseudo-classes inside CSS |
| `a >> b` | chain: search `b` inside `a` |
| `<sel> >> nth=0` | zero-based nth match |
| `<sel> >> visible=true` | only visible matches - better than nth when hidden duplicates exist |
| `id=myframe >>> .inner` | into an iframe |
| `\#submit` | a cell starting with `#` is a Robot comment: escape it or write `id=submit` / `css=#submit` |

`button=Calculate` (also `link=`, `input=`, `select=`, `textarea=`) is not Browser syntax. An ARIA snapshot line `button "Calculate"` is role + name, not a locator: use `role=button[name="Calculate"]`, `text=Calculate` or `css=button:text-is('Calculate')`.

## Strict mode

A locator that matches several elements fails ("strict mode violation ... resolved to 3 elements"). Narrow it - scope in a container, add `>> visible=true`, use role + name - rather than switching strict mode off (`Set Strict Mode    False`).

## Failure recipes

| Symptom | Fix |
|---|---|
| Click intercepted (overlay, sticky header, styled radio/checkbox) | find and close the overlay; `Scroll To Element    <sel>`; `Click With Options    <sel>    force=True` only as a last resort and never to reach hidden elements |
| Not interactable | `Wait For Elements State    <sel>    visible` (or `enabled`, `editable`), then act |
| Element detached / DOM rebuilt | `Wait For Elements State    <sel>    attached    timeout=10s`, then locate it again |
| Reactive form (Vue/React) ignores `Fill Text` | check `Get Property    <sel>    value`, then `Press Keys    <sel>    Tab` to commit |
| Slow page | `Set Browser Timeout    30s`, `Wait For Load State` |
| `Unsupported token` / invalid selector | a syntax error in the selector; fix it - waiting does not help |
| `Evaluate JavaScript` with only a JS body fails to parse | the first argument is a selector: `Evaluate JavaScript    ${None}    () => document.title` |

When a locator is rejected, try the same element in another syntax (`id=submit` -> `css=#submit`; `name=username` -> `css=[name='username']`), then re-inspect for `data-testid`, `aria-label` or role before guessing again.

Ported from the rf-mcp 0.36.0 locator guidance (`get_browser_locator_guidance`, hint rules) and checked against the Browser library cookbook suites in `tests/cookbook/`.
