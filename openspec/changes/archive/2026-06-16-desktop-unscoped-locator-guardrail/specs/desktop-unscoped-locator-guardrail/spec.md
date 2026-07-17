# Spec: desktop-unscoped-locator-guardrail

## ADDED Requirements

### Requirement: Unscoped desktop locators are refused pre-flight
For a desktop (PlatynUI) session, a `Query` or `Evaluate` keyword whose XPath argument is unscoped — starts with `//` or `descendant-or-self::`, or is a bare wildcard walk — SHALL be refused before the native evaluation is dispatched, with a structured error that names the locator, restates the scope-to-an-application rule, and offers an app-scoped rewrite. The error SHALL NOT reach the native AT-SPI walk.

#### Scenario: Unscoped Query refused
- **WHEN** a desktop session runs `Query  //control:Paragraph`
- **THEN** the response is `success: false` with a hint of type `unscoped_desktop_locator` that names `//control:Paragraph` and suggests `/app:*[@Name='<app>']//control:Paragraph`, and no native walk runs

#### Scenario: Bare wildcard walk refused
- **WHEN** a desktop session runs `Query  //*`
- **THEN** the call is refused pre-flight with the guidance hint

### Requirement: Scoped and discovery locators are allowed
Locators anchored to an application (`/app:*…`), relative locators (`control:…`, `.//…`, axis-prefixed), and pure scalar-aggregate expressions (`count(...)`, `string(...)`, `number(...)`, `boolean(...)`) SHALL NOT be refused.

#### Scenario: App-scoped Query allowed
- **WHEN** a desktop session runs `Query  /app:*[@Name='soffice']//control:Paragraph`
- **THEN** the guardrail does not refuse it

#### Scenario: count() discovery allowed
- **WHEN** a desktop session runs `Query  count(//control:Paragraph)`
- **THEN** the guardrail does not refuse it (counting is the sanctioned way to size a subtree before scoping)

#### Scenario: Relative query allowed
- **WHEN** a desktop session runs `Query  control:Button[@Name='OK']`
- **THEN** the guardrail does not refuse it

### Requirement: Explicit opt-out for desktop-wide search
When `ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED=1` or the session opt-out attribute is set, an unscoped desktop locator SHALL be allowed to run, with a single per-session warning rather than a refusal.

#### Scenario: Opt-out downgrades refusal to warning
- **WHEN** the opt-out is set and a desktop session runs `Query  //control:Paragraph`
- **THEN** the call proceeds and the response carries a one-time unscoped-locator warning

### Requirement: Non-desktop sessions unaffected
The guardrail SHALL apply only to desktop sessions; web, API, and mobile sessions SHALL be unaffected.

#### Scenario: Web session not guarded
- **WHEN** a web session executes a keyword with a `//`-rooted selector argument
- **THEN** the unscoped-locator guardrail does not refuse it
