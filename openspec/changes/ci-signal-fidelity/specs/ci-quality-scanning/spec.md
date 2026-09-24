## MODIFIED Requirements

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

## ADDED Requirements

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
