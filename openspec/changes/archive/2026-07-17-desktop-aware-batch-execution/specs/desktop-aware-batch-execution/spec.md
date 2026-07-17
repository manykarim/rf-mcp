# Spec: desktop-aware-batch-execution

## ADDED Requirements

### Requirement: execute_batch is available in the desktop tool profile
The `desktop_exec` tool profile SHALL include `execute_batch`, so that
desktop sessions on small-context models can collapse multi-step interaction
sequences into one round-trip. Activating `desktop_exec` SHALL leave
`execute_batch` callable.

#### Scenario: desktop profile exposes the batch tool
- **WHEN** the `desktop_exec` profile is activated
- **THEN** `execute_batch` is among the visible tools and a desktop-session batch call is not rejected on profile grounds

#### Scenario: profile stays within budget
- **WHEN** `desktop_exec` (now 7 tools) is validated against its 8192-context token budget
- **THEN** activation succeeds (directly or via the existing automatic description-mode fallback) without raising

### Requirement: PlatynUI element-resolution failures classify as element-not-found
The recovery error classifier SHALL classify PlatynUI descriptor-resolution
failures (error text containing `No UiNode found`, `ElementNotFoundError`, or
`UiNodeDescriptor`) as `ELEMENT_NOT_FOUND`, even though the message also
contains the word "timeout". Classification of existing browser error strings
SHALL be unchanged.

#### Scenario: the real PlatynUI error string
- **WHEN** the classifier receives `ElementNotFoundError: No UiNode found for UiNodeDescriptor query "//control:Button[@Name='7']" within timeout of 30 seconds.`
- **THEN** the classification is `ELEMENT_NOT_FOUND`, not `TimeoutException`

#### Scenario: browser errors unaffected
- **WHEN** the classifier receives existing browser-style errors (e.g. Playwright/Selenium timeout or element-not-found text)
- **THEN** each classifies exactly as before this change

### Requirement: Desktop sessions use desktop recovery strategies
Recovery strategy selection SHALL be platform-aware. For a failure in a
desktop session, the selected strategies SHALL execute only
desktop-meaningful actions (waiting via `Sleep`, re-activating the session's
current scoped root window); the browser-only recovery actions
(`Execute Javascript`, `Reload Page`, `Go Back`, `Handle Alert`) SHALL NOT be
executed in a desktop session. Strategy selection for web sessions SHALL be
unchanged.

#### Scenario: desktop element-not-found gets a desktop strategy
- **WHEN** a desktop batch step fails with an `ELEMENT_NOT_FOUND` classification and `on_failure="recover"`
- **THEN** the recovery attempt uses a desktop strategy (wait-then-retry, or activate-window-then-retry on escalation) and no browser-only keyword is dispatched

#### Scenario: window re-activation stays scoped
- **WHEN** the escalated desktop strategy runs and the session has a current root descriptor
- **THEN** `Activate Window` targets that root descriptor; **WHEN** no root is set, the strategy is skipped rather than issuing an unscoped activation

#### Scenario: web recovery unchanged
- **WHEN** a web-session failure is classified and a strategy is selected
- **THEN** the selected strategy and its actions are identical to the pre-change catalog

### Requirement: Descriptor-resolution timeout is capped during batch recovery retries
In a desktop session, recovery retries of a failed batch step SHALL run with
the PlatynUI descriptor-resolution timeout capped to a small bound (default
5 seconds, configurable via environment). The initial execution of each step
SHALL keep the native resolution budget. The prior timeout value SHALL be
restored after the retry completes, including when the retry raises, so
stepwise execution after the batch is unaffected.

#### Scenario: retries are bounded
- **WHEN** a desktop batch step fails element resolution and recovery retries it with default settings (2 attempts)
- **THEN** each retry's descriptor resolution is bounded by the cap, and the step's total wall time is materially below the ~93 s uncapped worst case (initial 30 s + bounded retries), keeping the 120 s batch budget survivable

#### Scenario: timeout restored on every exit path
- **WHEN** a capped retry finishes — by success, failure, or raised exception
- **THEN** the library's query timeout is restored to its prior value, and a subsequent stepwise `execute_step` observes the native 30 s behavior

#### Scenario: cap is desktop-only and soft
- **WHEN** the batch runs in a non-desktop session, or the PlatynUI library is not loaded
- **THEN** no timeout mutation is attempted and retry behavior is unchanged

### Requirement: Desktop batch recovery retries only provably-unfired inputs
In a desktop session, a failed batch step SHALL be retried only when the
failure indicates the input action never fired (element-not-found: descriptor
resolution precedes any pointer or keyboard action). Any other desktop step
failure SHALL be recorded immediately as a failure — equivalent to
`on_failure="stop"` — regardless of the requested `retry`/`recover` policy,
so a batch cannot blind-repeat clicks or keystrokes against an unknown
desktop state. The `execute_batch` documentation SHALL state this desktop
gate.

#### Scenario: unfired input is retried
- **WHEN** a desktop `Pointer Click` step fails with element-not-found under `on_failure="recover"`
- **THEN** the step is retried (with desktop recovery and the capped timeout)

#### Scenario: post-action failure is not blind-retried
- **WHEN** a desktop step fails with any non-element-not-found error (the action may have fired) under `on_failure="retry"` or `"recover"`
- **THEN** no retry occurs; the batch records the failure with diagnostics as the `stop` policy would

#### Scenario: gate is documented
- **WHEN** an agent reads the `execute_batch` tool description
- **THEN** it states that in desktop sessions only element-not-found failures are retried

### Requirement: Desktop init guidance steers agents toward execute_batch
The desktop session initialization guidance SHALL include a brief steer to
use `execute_batch` for multi-step interaction sequences. The full desktop
init cheat-sheet (keyword surface, locator crib) remains owned by the
`desktop-turn-economy-guidance` change; this requirement adds only the
batch-first pointer and SHALL NOT duplicate that content.

#### Scenario: init response mentions batching
- **WHEN** a desktop session is initialized
- **THEN** the returned guidance mentions `execute_batch` as the preferred way to run known multi-step sequences in one call
