## ADDED Requirements

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
