# frontend-dashboard Specification

## Purpose

The optional Django observability dashboard for rf-mcp ("RobotMCP Command Center", `src/robotmcp/frontend/`,
enabled via `--with-frontend` / `ROBOTMCP_ENABLE_FRONTEND=1` or `python -m robotmcp.frontend.devserver`).
It presents live Robot-Framework MCP sessions — session list, per-session summary/variables/steps, a
generated RF suite preview, and an event feed — for observation. This capability defines its security,
runtime, correctness, accessibility, and UX requirements.

## Requirements

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

### Requirement: The live event stream delivers events across event loops and degrades gracefully

The dashboard's event stream SHALL deliver events published from any thread/event loop (e.g. the MCP
server loop) to browser subscribers connected on the ASGI loop, streaming incrementally rather than
buffering, and SHALL clean up a subscriber's resources when the client disconnects. The client SHALL
fall back to polling recent events so a dropped stream degrades to stale-but-updating rather than a
frozen snapshot.

#### Scenario: an event published on one loop reaches a subscriber on another
- **WHEN** an event is published via `publish_sync` from the MCP loop while a browser is subscribed over SSE on the ASGI loop
- **THEN** the browser receives that event (cross-loop delivery is not a silent no-op)

#### Scenario: a disconnected client is cleaned up
- **WHEN** an SSE client disconnects
- **THEN** its subscriber queue is discarded and no thread is leaked

#### Scenario: a dead stream degrades instead of freezing
- **WHEN** the event stream is unavailable
- **THEN** the client keeps refreshing via a periodic poll rather than showing a permanently frozen view

### Requirement: The dashboard is available independently of MCP-session lifecycle

Under non-stdio transports the dashboard SHALL be served for the lifetime of the server process,
independent of individual MCP client sessions — it SHALL be reachable before any client connects and
SHALL NOT be torn down when a client session ends.

#### Scenario: the dashboard is up under HTTP transport
- **WHEN** the server runs with `--transport http --with-frontend`
- **THEN** the dashboard port is listening from startup and stays up across MCP client connect/disconnect

### Requirement: The dashboard renders session data faithfully

The dashboard SHALL render identifiers without blind character truncation (showing them in full or with
an explicit ellipsis), SHALL display user-defined variables regardless of naming convention (not hiding
ALL-CAPS names), SHALL NOT fabricate or duplicate library/event data, and SHALL NOT crash on the
zero-sessions path.

#### Scenario: identifiers are shown faithfully
- **WHEN** a session named `frontend-demo` is displayed
- **THEN** its name is shown in full (e.g. "Session frontend-demo"), not truncated to a misleading prefix

#### Scenario: user variables are visible
- **WHEN** a session defines ALL-CAPS variables (the Robot Framework convention, e.g. `CITY`, `BASE_URL`)
- **THEN** they appear in the Variables panel rather than being hidden

#### Scenario: library and event data are not fabricated or duplicated
- **WHEN** a session's libraries and lifecycle events are displayed
- **THEN** only libraries the session actually imported are shown (deduplicated), and a single lifecycle event is not shown twice

#### Scenario: the empty state does not crash
- **WHEN** there are no sessions (fresh install or last session closed)
- **THEN** the dashboard shows a clean empty state without throwing, and does not leave a stale session pane

### Requirement: The dashboard core flow is keyboard-operable with visible focus

The dashboard SHALL allow selecting a session using the keyboard alone (focusable session controls with
Enter/Space activation and appropriate roles/labels), SHALL render a visible focus indicator for
keyboard focus, and SHALL respect `prefers-reduced-motion` and allow touch scrolling of its panels.

#### Scenario: a session is selectable by keyboard
- **WHEN** a keyboard user focuses a session card and presses Enter or Space
- **THEN** that session is selected and its details are shown

#### Scenario: keyboard focus is visible
- **WHEN** an element receives keyboard focus
- **THEN** a visible focus indicator is shown

#### Scenario: motion and scrolling respect user constraints
- **WHEN** the user prefers reduced motion, or scrolls a panel on a touch device
- **THEN** animations are minimized and the panel scrolls rather than being locked

### Requirement: The dashboard surfaces failures and does not clip core controls

The dashboard SHALL indicate its live-connection state, SHALL degrade a single failed data request to
that region rather than blanking the whole view, and SHALL NOT clip primary controls (e.g. the suite
Generate button) or strand content behind a non-scrolling overflow.

#### Scenario: connection state is visible
- **WHEN** the live event stream connects, drops, or is unavailable
- **THEN** a connection indicator reflects Live / Reconnecting / Offline rather than silently showing frozen state

#### Scenario: one failed request does not blank the view
- **WHEN** one of the session-detail requests fails
- **THEN** the other regions still render, and the failure is surfaced rather than blanking the whole pane

#### Scenario: core controls are not clipped
- **WHEN** the dashboard is viewed on a narrow (mobile) viewport
- **THEN** the primary suite action remains visible/reachable, and the sidebar scrolls rather than stranding content
