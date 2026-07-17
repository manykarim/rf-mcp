# desktop-launch-env-generalization Specification

## Purpose
TBD - created by archiving change desktop-launch-env-generalization. Update Purpose after archive.
## Requirements
### Requirement: Desktop GUI launches are recognized by evidence, not an allowlist
The launch layer SHALL recognize a desktop GUI launch for any resolvable
executable started via the process keywords in a desktop session, not only the
fixed set of eight gnome binaries, while still excluding obvious non-GUI shell
utilities. Recognition drives snap-loader sanitization and the accessibility/
backend overlay, so a non-gnome AUT (LibreOffice, Firefox, a Qt app, mousepad)
receives the same launch hardening the gnome binaries already get.

#### Scenario: a non-allowlisted GUI binary is recognized in a desktop session
- **WHEN** a desktop session runs `Start Process    soffice    --writer` (or `mousepad`), a binary not in the known-gnome set
- **THEN** the launch is recognized as a desktop GUI launch and the sanitized child environment plus the accessibility/backend overlay are applied

#### Scenario: a plain subprocess is not treated as a GUI launch
- **WHEN** a desktop session runs `Start Process    bash    -c    echo hi` (or `python3 script.py`), a non-GUI utility on the deny-set
- **THEN** the launch is NOT recognized as a GUI launch and no accessibility/backend overlay or GUI sanitization is applied

#### Scenario: non-desktop sessions are unaffected
- **WHEN** a web/api session starts any process
- **THEN** the generalized GUI detection does not fire and no desktop overlay is applied

### Requirement: GUI launches receive a GTK_A11Y accessibility overlay
A recognized desktop GUI launch SHALL have `GTK_A11Y=atspi` present in its child
environment unless the variable is already set to a non-empty value, applied
alongside the existing DISPLAY / XDG_SESSION_TYPE=x11 / GDK_BACKEND=x11 /
QT_QPA_PLATFORM=xcb overlay, so a GTK or Qt AUT comes up with a populated AT-SPI
object tree regardless of the server process environment.

#### Scenario: a GTK AUT with no inherited GTK_A11Y comes up accessible
- **WHEN** the launch layer builds the child env for a recognized GUI AUT and the parent env has no `GTK_A11Y`
- **THEN** the child env contains `GTK_A11Y=atspi` (and the X11 backend pins), so the AUT exposes a non-empty accessibility tree

#### Scenario: an explicit operator GTK_A11Y value is preserved
- **WHEN** the parent env already sets `GTK_A11Y` to a non-empty value (e.g. `atspi` or a custom value)
- **THEN** the launch layer does not overwrite it

### Requirement: The launch layer reports which overrides it applied
The launch path SHALL report whether the AUT was recognized as a GUI launch and
which accessibility/backend overrides were applied, so an empty-object-tree
failure is diagnosable rather than silent (a desktop session cannot fall back to
`find_keywords`, which is useless for PlatynUI).

#### Scenario: applied overrides are surfaced for diagnosis
- **WHEN** a recognized GUI AUT is launched
- **THEN** the launch signal / session-state records the recognized binary and the applied override keys (e.g. `GTK_A11Y`, `GDK_BACKEND`, `XDG_SESSION_TYPE`, snap-stripped)

#### Scenario: an unrecognized launch is distinguishable from a hardened one
- **WHEN** a launch is not recognized as a GUI launch
- **THEN** the signal records that no GUI overlay was applied, so a later empty-tree symptom can be traced to the missing accessibility env
