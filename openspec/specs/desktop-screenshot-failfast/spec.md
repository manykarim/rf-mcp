# desktop-screenshot-failfast Specification

## Purpose
TBD - created by archiving change desktop-screenshot-failfast. Update Purpose after archive.
## Requirements
### Requirement: Take Screenshot with a filename in the descriptor slot fails fast
A desktop-session `Take Screenshot` SHALL be refused before native dispatch
when its descriptor slot — the first positional argument, or an explicit
`descriptor=` — is a bare image path, as a structured failure (not an unhandled
exception) whose hint names the `(descriptor, filename, rect)` signature and
shows both correct forms (`filename=<path>` named-only for a whole-desktop
screenshot; `<descriptor>    <path>` for an element screenshot). The step SHALL
NOT reach PlatynUI descriptor resolution (which would retry for ~30 s before
`ElementNotFoundError`, uncapped because desktop sessions skip timeout
injection).

#### Scenario: filename-only positional is refused with the signature hint
- **WHEN** a desktop session executes `Take Screenshot` with `["/artifacts/calc.png"]` as its only argument
- **THEN** the step is refused pre-dispatch with a structured failure whose hint states that the first positional binds to `descriptor`, names the `(descriptor, filename, rect)` signature, and shows the `filename=/artifacts/calc.png` and `<descriptor>    /artifacts/calc.png` corrections — with no 30 s descriptor-resolution wait

#### Scenario: correct forms proceed unchanged
- **WHEN** a desktop session executes `Take Screenshot` with a descriptor followed by a path positional, or with only a named `filename=<path>` argument, or with `EMBED`
- **THEN** the guard does not fire and the keyword dispatches as before

#### Scenario: non-desktop sessions are unaffected
- **WHEN** a web session executes a `Take Screenshot` keyword with a bare path positional (valid in Browser/SeleniumLibrary)
- **THEN** the guard does not fire

### Requirement: Linux control:Window locators fail fast with Frame guidance
On Linux desktop sessions, the executor SHALL refuse before native dispatch a
descriptor/XPath argument containing the `control:Window` role token, when
passed to a keyword that resolves it against the accessibility tree (`Query`,
`Evaluate`, `Set Root`, `Get Attribute`, and pointer/keyboard interaction
keywords), with a structured failure whose hint states that AT-SPI top-level windows have
role Frame (`control:Frame`), and offers the submitted locator rewritten with
`control:Window` replaced by `control:Frame`. The locator SHALL NOT be silently
rewritten — the recorded step must match what executed. On non-Linux platforms
(where `control:Window` is correct under UIA) the guard SHALL NOT fire.

#### Scenario: control:Window query on Linux is refused with a rewrite hint
- **WHEN** a Linux desktop session executes `Query` (or `Set Root`, `Pointer Click`, …) with `/app:*[@Name='gnome-calculator']//control:Window`
- **THEN** the step is refused pre-dispatch with a hint stating the Frame-vs-Window AT-SPI fact and offering `/app:*[@Name='gnome-calculator']//control:Frame` — with no 30 s descriptor-resolution wait

#### Scenario: control:Frame and non-Linux platforms proceed
- **WHEN** the locator uses `control:Frame`, or the platform is Windows
- **THEN** the guard does not fire and the keyword dispatches as before

#### Scenario: the locator is not silently mutated
- **WHEN** the guard refuses a `control:Window` locator
- **THEN** no keyword executes with a modified locator; the agent must resubmit the corrected step itself

### Requirement: Both guards honor an explicit opt-out consistent with the unscoped-locator guard
Each guard SHALL honor a deliberate escape hatch mirroring
`ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED`: an environment variable
(`ROBOTMCP_PLATYNUI_ALLOW_PATH_DESCRIPTOR` for the screenshot guard,
`ROBOTMCP_PLATYNUI_ALLOW_CONTROL_WINDOW` for the window-role guard) or the
equivalent session flag. When opted in, the step SHALL proceed and the refusal
SHALL be downgraded to a warning surfaced at most once per session.

#### Scenario: opt-in proceeds with a one-time warning
- **WHEN** the relevant opt-out env var (or session flag) is set and a step matches a guard
- **THEN** the step dispatches, the response carries a warning hint explaining the risk, and subsequent matching steps in the same session do not repeat the warning

#### Scenario: without opt-in the refusal names the escape hatch
- **WHEN** a step is refused by either guard without the opt-out set
- **THEN** the hint mentions the corresponding opt-out variable so a deliberate use is still possible

