## ADDED Requirements

### Requirement: Expected keyword failures do not dump tracebacks at default level

When a keyword execution fails for a reason that is caught and returned to the
client as a structured error (e.g. an HTTP error status, an unknown keyword, an
unresolved variable), rf-mcp SHALL log a single-line WARNING/ERROR summary at the
default level and SHALL emit the full Python traceback only at DEBUG level.

#### Scenario: a 404 keyword failure produces a one-line summary, not a stack dump
- **WHEN** an agent runs a keyword that fails with a handled error and the log level is the default
- **THEN** stderr contains a concise summary of the failure but not a multi-frame Python traceback, and the failure is still returned to the client as a structured error

### Requirement: Lazy RF-context bootstrap paths log at their true severity

Messages describing a normal try→fallback during RF-context bootstrap that
subsequently succeeds SHALL NOT be logged as WARNING/ERROR at the default level.

#### Scenario: recoverable context registration is not logged as a failure
- **WHEN** a fresh session registers a library and the first registration attempt falls back to an on-demand import that succeeds
- **THEN** the default-level logs do not contain a WARNING/ERROR implying the registration failed
- **WHEN** the on-demand import itself fails
- **THEN** a WARNING/ERROR is still logged

### Requirement: Page-source inspection is gated on a loaded web library

The page-source service SHALL short-circuit for sessions with no web/browser
library loaded (e.g. API/Requests-only or desktop sessions) and SHALL NOT attempt
browser DOM keywords for them.

#### Scenario: get_session_state on an API session does not cascade DOM keywords
- **WHEN** `get_session_state` (including with no sections) is called on a Requests-only session
- **THEN** the page-source service returns a "no DOM (non-web session)" result without invoking browser page-source keywords, and no DOM-keyword failure tracebacks are produced

### Requirement: The assignment heuristic recognizes RequestsLibrary response keywords

The return-value assignment heuristic SHALL treat RequestsLibrary request keywords
(`GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS On Session` and the bare
`GET/POST/…`) as value-returning, and SHALL NOT warn when their result is assigned.

#### Scenario: capturing a response does not trigger a false warning
- **WHEN** a step assigns the result of `GET On Session` to a variable
- **THEN** no "may not return a useful value" warning is emitted for that step

### Requirement: Common RequestsLibrary name mistake is normalized

When a session is initialized requesting the library name `Requests`, rf-mcp SHALL
resolve it to `RequestsLibrary` and SHALL log the correction at WARNING level
rather than surfacing an ERROR-level import failure.

#### Scenario: `Requests` resolves to `RequestsLibrary`
- **WHEN** a session init requests library `Requests`
- **THEN** `RequestsLibrary` is loaded and a WARNING-level correction is logged, with no ERROR-level ModuleNotFoundError for `Requests`

### Requirement: Requests guidance covers JSON request-body construction

The requests guidance returned by `get_locator_guidance(library="requests")` SHALL
include how to construct a JSON request body, including an inline-eval form and the
rule to define the body before the request.

#### Scenario: the cookbook shows the JSON-body pattern
- **WHEN** an agent requests the requests guidance
- **THEN** the guidance includes the `json=${{ {...} }}` inline-eval body form and the define-body-before-POST ordering rule
