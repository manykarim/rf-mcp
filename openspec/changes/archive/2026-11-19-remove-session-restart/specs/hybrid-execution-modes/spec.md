# Spec Delta

## MODIFIED Requirements

### Requirement: Sessions work in a layered environment
REPL and debug sessions SHALL work when robotcode is layered onto a project interpreter: a session started by one call SHALL accept `send`, `status`, `result`, `interrupt` and `stop` from later calls, each of which runs as its own process.

#### Scenario: REPL across separate calls
- **WHEN** a global rf-mcp starts `rf_repl` for a root whose `.venv` holds RequestsLibrary but no robotcode, then sends `Import Library    RequestsLibrary` and `${r}=    Evaluate    6*7`, then stops the session
- **THEN** both sends pass, the second reports `${r} = 42`, and no session for that root remains open
