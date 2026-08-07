## ADDED Requirements

### Requirement: The dashboard runs with a safe-by-default production posture

The dashboard SHALL default to `DEBUG=False` and SHALL scope `ALLOWED_HOSTS` to the configured host
plus loopback (with an explicit environment override) rather than `*`, and it SHALL refuse to start
with debug enabled while bound to a non-loopback interface. Debug mode SHALL require an explicit opt-in.

#### Scenario: debug is off unless explicitly enabled
- **WHEN** the dashboard starts without an explicit debug opt-in
- **THEN** Django `DEBUG` is `False`, so an unhandled error does not return SECRET_KEY, settings, environment variables, or source to the client

#### Scenario: host validation is scoped
- **WHEN** the dashboard resolves `ALLOWED_HOSTS`
- **THEN** it uses the configured host plus loopback (and any explicit env override), not `*`

#### Scenario: debug on a public bind is refused
- **WHEN** debug is enabled and the frontend is bound to a non-loopback interface
- **THEN** startup fails with a clear error rather than serving debug pages to the network

### Requirement: The dashboard exposes no unauthenticated mutating or code-execution endpoint

The dashboard SHALL NOT expose an unauthenticated endpoint that executes arbitrary keywords or
otherwise mutates server state. The arbitrary-keyword execution route SHALL be removed (it has no UI
consumer); any future mutating endpoint SHALL require authentication and SHALL default-refuse on a
non-loopback bind.

#### Scenario: the execute-keyword route is not reachable
- **WHEN** a client POSTs to the former execute-keyword URL
- **THEN** the request is not routed to a keyword executor (the route does not exist)

### Requirement: The dashboard escapes untrusted data rendered in the browser

The dashboard SHALL render session-, variable-, step-, and event-derived text (which can contain
content scraped from the page under test) as text nodes / safe DOM construction, never by interpolating
it into `innerHTML`. Only static, developer-authored strings MAY be assigned via `innerHTML`.

#### Scenario: attacker-controlled text does not execute
- **WHEN** a session variable name or value contains HTML/script markup
- **THEN** it is displayed as literal text and no script executes in the dashboard origin

### Requirement: The dashboard constrains its client-side assets

The dashboard SHALL emit a Content-Security-Policy and SHALL load any third-party client assets with
subresource-integrity pinning or serve them locally, so a compromised or unavailable CDN cannot inject
script or silently break the UI.

#### Scenario: third-party assets are integrity-pinned or local
- **WHEN** the dashboard page loads its icon/syntax-highlighting assets
- **THEN** they are either served from the app's own origin or pinned with an integrity hash, and a Content-Security-Policy restricts script sources
