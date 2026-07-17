# visual-inspection-guidance Specification

## Purpose
TBD - created by archiving change visual-inspection-guidance. Update Purpose after archive.
## Requirements
### Requirement: Screenshot responses are token-cheap by default and advertise the artifact
A successful screenshot step SHALL return the saved image's absolute path plus a
short visual-validation hint, and SHALL NOT embed image bytes in the default
response. This keeps the default path a few tokens (not ~1,000+), so a multimodal
agent spends image tokens only when it chooses to read the file, and a text-only
agent is unaffected.

#### Scenario: a successful screenshot advertises its path + hint
- **WHEN** a desktop or web screenshot keyword succeeds
- **THEN** the response includes the saved image's absolute path and a one-line hint that it can be read to validate visually (naming the vision-only case types and "prefer Get Text when the value is in the DOM"), with no image content block

#### Scenario: non-screenshot steps are unaffected
- **WHEN** a non-screenshot keyword runs
- **THEN** no screenshot path or visual hint is added

### Requirement: A visual_validation guidance topic teaches when vision beats the DOM
`get_locator_guidance` SHALL accept `library="visual"` (and the alias `screenshot`)
and return guidance covering the vision-only validation cases, the dual read-back
pattern, and the multimodal + file-access caveats — so an agent learns the
capability exists and applies it only where structured trees cannot answer.

#### Scenario: the visual topic returns the case guidance
- **WHEN** `get_locator_guidance(library="visual")` is called
- **THEN** it returns the vision-only case categories (canvas/image text, layout/overlap, obscured elements, color, charts, desktop text content, validation gestalt) and the dual read-back rule (assert with Get Text first when a node exposes the value; screenshot to confirm; screenshot-primary only when no node carries it)

#### Scenario: the guidance names the gating caveats
- **WHEN** the visual guidance is returned
- **THEN** it states that visual validation requires a multimodal driving model and (for the default path) file access to the screenshot directory, and that vision is a fallback/complement oracle — not a deterministic CI primary

### Requirement: An opt-in visual_check tool returns an image only when explicitly requested
A dedicated `visual_check` tool SHALL capture a screenshot across any of the four UI
libraries and, by default, return only the saved path as text; it SHALL return a
multimodal image content block ONLY when the caller explicitly requests it and the
deployment mode permits, so text-only driving models are never sent unsupported
image content.

#### Scenario: default returns a path, not an image
- **WHEN** `visual_check` is called without requesting an image
- **THEN** it saves the screenshot and returns the path (and size/dimensions) as text, with no image content block

#### Scenario: image returned only on explicit request
- **WHEN** `visual_check` is called with the image-return option and `ROBOTMCP_SCREENSHOT_MODE` permits it
- **THEN** the response includes an image content block for a multimodal model, alongside the saved path

#### Scenario: text-only deployment forces the path fallback
- **WHEN** `ROBOTMCP_SCREENSHOT_MODE=file` (or the mode disallows image content)
- **THEN** even an image-return request yields only the path text, so a text-only driver is never sent an image block

#### Scenario: a failed capture degrades cleanly
- **WHEN** the screenshot cannot be captured or the file is missing
- **THEN** `visual_check` returns the existing evidence-missing hint and does not raise

