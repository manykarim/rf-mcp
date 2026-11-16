## ADDED Requirements

### Requirement: Hints link to skill sections
A hint rule, built-in or from `hints.toml`, SHALL be able to name skill URIs in order of preference. An attached hint SHALL carry the first of those URIs that the server serves for the call's project root, as `skill`. When none of them is served, the hint SHALL carry no `skill`. Hints returned by `rf_diagnose` SHALL follow the same rule.

#### Scenario: Official skill served
- **WHEN** a `Browser` keyword fails with `strict mode violation`, the built-in rule links `skill://rf-browser/references/locators.md` and then `skill://rf-locators/references/browser.md`, and the official skills are served
- **THEN** the attached hint carries `skill` = `skill://rf-browser/references/locators.md`

#### Scenario: Official skills absent
- **WHEN** the same failure happens and the official skills package is not installed
- **THEN** the attached hint carries `skill` = `skill://rf-locators/references/browser.md`

#### Scenario: Project rule links a company skill
- **WHEN** `.robotframework/hints.toml` defines a rule for `CorpAuthExpiredError` with `skills = ["skill://acme-auth/references/tokens.md"]`, that skill is served from `--skills-dir`, and a keyword fails with that error
- **THEN** the attached hint carries `skill` = `skill://acme-auth/references/tokens.md`

#### Scenario: Linked skill not served
- **WHEN** a rule links only `skill://acme-auth/SKILL.md` and no served skill has that name
- **THEN** the attached hint carries no `skill`, and its other fields are unchanged

#### Scenario: Built-in links stay valid
- **WHEN** the test suite runs against the pinned official skills package and the built-in skills
- **THEN** every skill URI named by a built-in hint rule resolves to a file in at least one of them, and a link that resolves nowhere fails the build
