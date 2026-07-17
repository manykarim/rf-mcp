# Spec: api-cookbook

## ADDED Requirements

### Requirement: get_locator_guidance provides a proactive RequestsLibrary cookbook
The `get_locator_guidance` tool SHALL accept `library` values `requests`,
`requestslibrary`, and `api` (case-insensitive) and return a RequestsLibrary
request/response cookbook in the same payload shape as the other supported
libraries (`{success, library, tips/warnings/examples}`), so an agent can obtain
the request/response patterns proactively — before hand-rolling `Evaluate`-based
assertions — the same way web flows pull locator guidance.

#### Scenario: requests library returns the cookbook
- **WHEN** `get_locator_guidance(library="requests")` is called
- **THEN** it returns `success: true`, `library: "RequestsLibrary"`, and cookbook content covering session setup, response-field access, status assertion, JSON body/headers, auth token header, and non-2xx assertion

#### Scenario: aliases resolve to the cookbook
- **WHEN** `get_locator_guidance` is called with `library` = `api`, `requestslibrary`, or a differently-cased `Requests`
- **THEN** the same RequestsLibrary cookbook is returned

#### Scenario: other libraries and errors are unaffected
- **WHEN** `get_locator_guidance` is called for Browser/Selenium/Appium/PlatynUI, or for an unsupported library
- **THEN** the existing per-library guidance (or the existing unsupported-library error payload) is returned unchanged

### Requirement: The cookbook covers the response-access patterns that cause API turn waste
The RequestsLibrary cookbook SHALL include the specific recipes that the evals
identified as the dominant API turn sink: reading a JSON field via
`${resp.json()["field"]}`; the rule that inside `Evaluate` the bare `$resp` form
is used while `${resp.json()}` is used elsewhere; the native `Status Should Be`
status assertion as the alternative to `Evaluate`-based status comparison; the
`Cookie: token=<token>` auth header pattern; and `expected_status=` for
asserting non-2xx responses without raising.

#### Scenario: the cookbook names the native status assertion
- **WHEN** the RequestsLibrary cookbook is returned
- **THEN** it includes `Status Should Be` (with the response object) as the way to assert status, explicitly instead of an `Evaluate` equality on the status code

#### Scenario: the cookbook states the Evaluate variable-syntax rule
- **WHEN** the RequestsLibrary cookbook is returned
- **THEN** it states that `$resp` (bare) is used inside `Evaluate` and `${resp.json()}` is used outside it

#### Scenario: the cookbook covers auth and non-2xx assertions
- **WHEN** the RequestsLibrary cookbook is returned
- **THEN** it includes the `Cookie: token=<token>` header recipe and the `expected_status=` recipe for asserting error responses (e.g. a deleted resource returning 404)

### Requirement: Proactive and reactive RequestsLibrary guidance stay consistent
The proactive cookbook and the existing reactive on-failure hints (`utils/hints.py`)
SHALL share their common recipe text so the two cannot drift, and the reactive
on-failure hints SHALL remain in place as the safety net.

#### Scenario: shared recipes do not drift
- **WHEN** a recipe (e.g. Create Session + On Session, or the Evaluate `$var` rule) appears in both the proactive cookbook and a reactive hint
- **THEN** both render the same underlying text (a single shared source), and the existing reactive hint behavior is unchanged
