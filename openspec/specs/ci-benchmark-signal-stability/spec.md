# ci-benchmark-signal-stability Specification

## Purpose

Governs when a performance measurement or an environment failure is allowed to fail a
build, so that a red check means something a contributor can act on.

Every one of the 21 dependency pull requests merged on 2026-09-24 was initially red, and
not one needed a code change. Telling those apart from real breakage took per-failure log
forensics each time - the cost of a check that cries wolf is paid by every contributor who
has to investigate it.

Two causes recurred. Ratio assertions compared durations too small for the timing source
to resolve: one failed at "2.52x > 2.0" over a 0.6 microsecond difference, and the same
comparison measured 1.03x once the sample was taken as a best-of-N minimum - the true
overhead was 3%, not 152%. And browser-startup failures surfaced as assertions about page
content, so "Chrome did not launch" read as a product defect in page-source routing.

An absolute-time floor alone is not sufficient and this capability does not settle for
one: a floor guards only the denominator of a ratio, and a measurement that already had
one still failed at 65.76x when interference landed in the numerator.

## Requirements

### Requirement: A performance assertion only fails on a measurement it can resolve
A performance test SHALL NOT fail a build on a ratio between measurements that are too
small for the timing source to resolve reliably. A ratio assertion SHALL apply only when
the measured durations meet a declared absolute floor; below that floor the measurement
SHALL be recorded and reported without failing.

#### Scenario: a sub-resolution ratio is recorded, not asserted
- **WHEN** both sides of a ratio comparison are below the declared absolute floor
- **THEN** the measurement is recorded and reported, and the test does not fail on the
  ratio, because at that scale the ratio reflects scheduler noise rather than the code

#### Scenario: a real regression still fails
- **WHEN** the measured durations exceed the absolute floor and the ratio exceeds its limit
- **THEN** the test fails, because the measurement is large enough for the ratio to mean
  something

#### Scenario: the floor is stated, not implicit
- **WHEN** a ratio assertion is introduced
- **THEN** the absolute floor it requires is declared with it, so a later reader can tell
  a deliberately-skipped comparison from an overlooked one

### Requirement: Environment failures are distinguishable from product failures
A test SHALL report a failure to start or reach required external infrastructure as an
environment failure, distinctly from an assertion about product behaviour, so that a
reader can tell "the browser would not launch" from "the product returned the wrong
answer" without reading the stack trace.

#### Scenario: a browser that will not start is reported as such
- **WHEN** a test cannot obtain a browser session because the browser failed to launch
- **THEN** the failure identifies the environment as the cause, rather than surfacing as
  an assertion about the page content

#### Scenario: a genuine wrong answer is still an assertion failure
- **WHEN** the infrastructure is available and the product returns the wrong result
- **THEN** the test fails as an assertion about product behaviour
