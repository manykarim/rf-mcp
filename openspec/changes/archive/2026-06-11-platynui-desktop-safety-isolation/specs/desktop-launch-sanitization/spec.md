## ADDED Requirements

### Requirement: Sanitize the child environment for desktop AUT launches

The system SHALL sanitize, on the **child** launch environment only (never the
server's own environment), the package-rooted loader/module/data variables that
break a desktop GUI app launched from a desktop session (e.g. via
`Process.Start Process`) — removing `/snap/`-rooted path segments from
`LD_LIBRARY_PATH`, `LD_PRELOAD`, `GTK_PATH`, `GTK_EXE_PREFIX`,
`GIO_MODULE_DIR`, `GIO_EXTRA_MODULES`, `GSETTINGS_SCHEMA_DIR`, `QT_PLUGIN_PATH`,
`XDG_DATA_DIRS`, `FONTCONFIG_FILE`, `FONTCONFIG_PATH`, and `LOCPATH` — by
filtering individual path segments under known package roots rather than
clearing whole variables.

#### Scenario: Snap-rooted loader segments removed for the AUT
- **WHEN** rf-mcp launches a known desktop GUI app while the server environment
  contains snap-rooted entries in `LD_LIBRARY_PATH`/`GTK_PATH`/`GIO_MODULE_DIR`/
  `XDG_DATA_DIRS`
- **THEN** the launched child does not inherit the snap-rooted segments,
  avoiding the `__libc_pthread_init ... GLIBC_PRIVATE` symbol lookup failure,
  while non-snap segments of those variables are preserved

#### Scenario: Non-contaminated launches are unchanged
- **WHEN** the server environment has no package-rooted contamination
- **THEN** the launch environment is passed through unchanged apart from the
  display variables below

#### Scenario: Snap-confined AUT keeps its own snap roots
- **WHEN** the AUT being launched is itself a snap-confined binary that needs
  its snap's roots
- **THEN** the sanitizer preserves that snap's variables (it does not strip the
  roots the AUT requires), and a `--no-sanitize` escape hatch is available

### Requirement: Desktop AUT inherits the bound isolated display

The system SHALL ensure a desktop AUT launched from a desktop session inherits
the session's display environment (`DISPLAY`, `XDG_SESSION_TYPE=x11`,
`GDK_BACKEND=x11`, `WAYLAND_DISPLAY` unset) so the app appears on the same
display the runtime is bound to.

#### Scenario: Launched app appears on the bound display
- **WHEN** a desktop session bound to an isolated display launches its AUT
- **THEN** the AUT's window is created on that isolated display, not on a
  different/host display

### Requirement: Diagnose immediate-exit GUI launches

The system SHALL detect when a launched desktop GUI app exits immediately and
surface a diagnostic (captured stderr + the likely env-contamination cause)
instead of proceeding as if the app were running.

#### Scenario: Immediate exit reported with cause
- **WHEN** a launched desktop GUI app exits within a short window with a
  dynamic-loader/symbol error on stderr
- **THEN** the step result reports the immediate exit and points to the
  environment-contamination cause and mitigation
