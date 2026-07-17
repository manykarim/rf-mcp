# Spec: desktop-aut-process-lineage

## ADDED Requirements

### Requirement: AUT scope verified by process lineage
The focus gate's AUT process-identity check SHALL treat a resolved target as in scope when the target pid equals the launched pid, the target's parent chain reaches the launched pid, or the target's process session id matches the session id captured at launch — not by bare pid equality.

#### Scenario: Wrapper-script launch does not warn
- **WHEN** the AUT was launched as `bash wrapper.sh` (captured pid = bash) and the resolved target's application pid is the wrapper's live child
- **THEN** no process-identity warning is emitted

#### Scenario: Daemonized AUT does not warn
- **WHEN** the AUT process re-parented to init after launch (e.g. LibreOffice's oosplash fork) but shares the session id captured at launch
- **THEN** no process-identity warning is emitted

#### Scenario: Single-instance handoff does not warn
- **WHEN** a second launch's files are served by the original AUT process (new launcher pid, old application pid, same session id)
- **THEN** no process-identity warning is emitted

### Requirement: Confirmed foreign targets still warn
When both the target's lineage signals and the launch records are resolvable and show no relation (different pid, no ancestor link, different session id), the warning SHALL fire and SHALL state the checked lineage (pids and session ids).

#### Scenario: Foreign desktop application
- **WHEN** the resolved target belongs to a process from a different session with no ancestor relation to the launched AUT
- **THEN** a warning names both pids and session ids and says commands may be going to a different application

### Requirement: Indeterminate lineage is silent
When lineage cannot be established (launcher exited and reads fail, or no `/proc` semantics on the platform), no warning SHALL be emitted.

#### Scenario: Dead launcher, unreadable target
- **WHEN** the launched process has exited and the target's parent/session reads fail
- **THEN** the step proceeds with no process-identity warning
