# ci-quality-scanning Specification

## Purpose

Governs continuous code-quality and dependency-vulnerability analysis for rf-mcp, using
only open-source tooling. It replaces a SonarCloud integration that had been failing on
`main` for consecutive commits without anyone acting on it - a permanently-red check trains
contributors to ignore the check column - and closes a complete absence of dependency
scanning. The defining constraint is that findings must be VISIBLE without being
BLOCKING: the analysis reports through a digest, a downloadable artifact and code
scanning, and never fails the build on a finding, while still failing when an analyzer
could not run at all.

## Requirements

### Requirement: Analysis reports findings without failing the build
The quality analysis SHALL NOT cause any failed check on a pull request as a result of
findings, including checks raised downstream by the code-scanning service rather than by
the job itself. The job SHALL complete successfully regardless of how many findings the
analyzers report, so that a finding never blocks a merge or marks the run as broken. The
job SHALL NOT rely on `continue-on-error`, which leaves a failed job and a red
"some checks were not successful" state on the pull request. Only an infrastructure
failure - an analyzer that cannot be installed or run at all - SHALL fail the job.

#### Scenario: findings leave the check green
- **WHEN** the analyzers report findings of any severity
- **THEN** the job concludes successfully and the pull request shows no failed check for it

#### Scenario: a broken analyzer is still reported
- **WHEN** an analyzer cannot be installed or exits for a reason other than having found issues
- **THEN** the job fails, because the absence of analysis must not look like a clean result

#### Scenario: no finding blocks a merge
- **WHEN** the repository's merge requirements are evaluated for a pull request
- **THEN** the quality job's findings do not prevent the merge

#### Scenario: a downstream code-scanning check does not go red either
- **WHEN** the published findings cause the code-scanning service to raise its own check
  on the pull request
- **THEN** that check does not report failure on account of findings, so "the analysis
  never fails the build" holds for every check the analysis produces, not only the job

#### Scenario: touching code near an existing finding is not reported as a new problem
- **WHEN** a pull request edits or relocates a line that already carried a finding, and
  the project-wide count for that rule does not increase
- **THEN** no check fails, because the contributor introduced nothing

### Requirement: Every run prints a short human-readable digest
The job SHALL print a compact digest of the findings to the job log AND write the same
digest to the run summary, covering at minimum: code-quality finding count, security
finding count with the number at the highest severity, dependency advisory count with the
number of affected packages, and the count of files at the maintainability floor. The
digest SHALL state where the full reports and the published findings can be found.

#### Scenario: the digest is visible without opening a report
- **WHEN** a run completes
- **THEN** the log and the run summary each show the counts for code quality, security, dependencies and maintainability

#### Scenario: the digest points at the detail
- **WHEN** a reader wants more than the counts
- **THEN** the digest names the artifact holding the full reports and states that findings are published to code scanning

#### Scenario: a clean run is unambiguous
- **WHEN** an analyzer reports no findings
- **THEN** the digest shows zero for that analyzer rather than omitting the line

### Requirement: Findings are published in a machine-readable form
The job SHALL emit machine-readable reports for every analyzer, upload them as a
downloadable build artifact, and publish the SARIF-capable results to GitHub code
scanning. Each SARIF upload SHALL carry a distinct category so that one analyzer's
results do not replace another's.

#### Scenario: reports are downloadable
- **WHEN** a run completes
- **THEN** a build artifact contains the machine-readable report for each analyzer

#### Scenario: findings appear in code scanning
- **WHEN** an analyzer that emits SARIF reports findings
- **THEN** those findings appear as code scanning results for the commit

#### Scenario: analyzers do not overwrite each other
- **WHEN** more than one SARIF file is published in the same run
- **THEN** each analyzer's results remain separately visible

### Requirement: Code quality and code security are analysed
The job SHALL analyse the project source for code-quality and code-security issues using
a configured, version-pinned rule set, and the rule set SHALL be declared in the
repository rather than left to the analyzer's evolving defaults.

#### Scenario: the rule set is explicit
- **WHEN** the analyzer runs
- **THEN** it uses the rule set declared in the repository, so an upstream change to the analyzer's defaults cannot silently change the finding volume

#### Scenario: security findings are identifiable
- **WHEN** the analysis reports findings
- **THEN** security-category findings can be distinguished from style and correctness findings

### Requirement: Dependency vulnerabilities are analysed from the lockfile
The job SHALL scan the project's resolved dependency lockfile for known vulnerabilities
and report the advisories found, so that transitive dependencies are covered at the exact
versions that are installed.

#### Scenario: transitive dependencies are covered
- **WHEN** a vulnerable package is present only as a transitive dependency in the lockfile
- **THEN** its advisories are reported

#### Scenario: an advisory with no fix is still reported
- **WHEN** an advisory has no published fixed version
- **THEN** it is still reported rather than omitted for being unactionable

### Requirement: Maintainability hotspots are reported
The job SHALL report which source files sit at the maintainability floor, so the small set
of files that dominate the project's maintenance cost stays visible.

#### Scenario: floor files are listed
- **WHEN** the analysis runs
- **THEN** the digest reports how many files are at the maintainability floor and the reports identify them

### Requirement: The repository has one analysis signal
Dependency security alerting SHALL be enabled on the repository, and the superseded
third-party analysis integration SHALL be removed, so contributors have a single place to
look rather than a working signal beside a permanently failing one.

#### Scenario: dependency alerts are active
- **WHEN** a new advisory affects a dependency of this repository
- **THEN** the repository raises an alert without requiring a CI run

#### Scenario: the superseded integration no longer reports
- **WHEN** a pull request is opened after this change
- **THEN** the retired analysis integration does not contribute a check result

### Requirement: Published severity is assigned deliberately, not inherited
The analysis SHALL assign the severity it publishes for each finding from a declared
rule set rather than accepting the level the analyzer emits, because an analyzer that
labels every finding identically carries no severity information. The declared
high-severity set SHALL be recorded in the repository alongside the rule selection, and
SHALL be validated against an independent tool's severity rating.

#### Scenario: a uniform analyzer level is not treated as severity
- **WHEN** an analyzer marks every finding at one level regardless of what it found
- **THEN** the published severity comes from the declared rule set instead, so a
  low-severity style finding is not published as an error

#### Scenario: genuinely high-severity findings stay prominent
- **WHEN** a finding belongs to the declared high-severity rule set
- **THEN** it is published at error severity and appears in the digest's high count

#### Scenario: the declared set is justified, not asserted
- **WHEN** the high-severity rule set is introduced or changed
- **THEN** its membership is checked against an independent tool's high-severity
  findings for the same source, and the comparison is recorded
